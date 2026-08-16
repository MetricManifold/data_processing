#include "palmieri_initializer.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool uniform53_golden_vectors() {
    constexpr std::array<std::uint64_t, 4> expected_raw = {
        0xacf1a4f6dd62376full,
        0xda7817282b5b572full,
        0x9f50d717c235da9aull,
        0x6bbddc9ddcc34f7aull,
    };
    constexpr std::array<double, 4> expected_u = {
        0x1.59e349edbac46p-1,
        0x1.b4f02e5056b6ap-1,
        0x1.3ea1ae2f846bbp-1,
        0x1.aef77277730d2p-2,
    };
    std::mt19937_64 rng(1729);
    bool pass = pf::palmieri_u01_from_u64(0) == 0.0;
    pass = pass && pf::palmieri_u01_from_u64(UINT64_MAX) ==
                       0x1.fffffffffffffp-1;
    for (std::size_t i = 0; i < expected_raw.size(); ++i) {
        const std::uint64_t raw = rng();
        pass = pass && raw == expected_raw[i];
        pass = pass && pf::palmieri_u01_from_u64(raw) == expected_u[i];
    }
    std::printf("uniform53_golden=%d\n", pass);
    return pass;
}

bool write_bytes(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), (std::streamsize)bytes.size());
    return (bool)out;
}

bool rejected(const std::filesystem::path& path, const std::string& bytes,
              int n, double side, double radius) {
    if (!write_bytes(path, bytes)) return false;
    std::vector<float> x, y;
    std::string error;
    return !pf::palmieri_read_centres_csv(path.string(), n, side, radius,
                                           &x, &y, nullptr, &error) &&
           !error.empty();
}

bool malformed_cases(const std::filesystem::path& root) {
    const std::string valid = "global_id,x,y\n0,50,50\n1,10,10\n";
    bool pass = true;
    pass = pass && rejected(root / "header.csv",
        "x,global_id,y\n0,50,50\n1,10,10\n", 2, 100, 10);
    pass = pass && rejected(root / "id.csv",
        "global_id,x,y\n1,50,50\n0,10,10\n", 2, 100, 10);
    pass = pass && rejected(root / "short.csv",
        "global_id,x,y\n0,50,50\n", 2, 100, 10);
    pass = pass && rejected(root / "long.csv",
        valid + "2,80,80\n", 2, 100, 10);
    pass = pass && rejected(root / "bounds.csv",
        "global_id,x,y\n0,50,50\n1,100,10\n", 2, 100, 10);
    pass = pass && rejected(root / "centre.csv",
        "global_id,x,y\n0,49,50\n1,10,10\n", 2, 100, 10);
    pass = pass && rejected(root / "separation.csv",
        "global_id,x,y\n0,50,50\n1,55,50\n", 2, 100, 10);
    pass = pass && rejected(root / "periodic_separation.csv",
        "global_id,x,y\n0,50,50\n1,1,1\n2,99,1\n", 3, 100, 3);
    pass = pass && rejected(root / "nonfinite.csv",
        "global_id,x,y\n0,50,50\n1,nan,10\n", 2, 100, 10);
    pass = pass && rejected(root / "numeric.csv",
        "global_id,x,y\n0,50,50\n1,10,10junk\n", 2, 100, 10);
    pass = pass && rejected(root / "crlf.csv",
        "global_id,x,y\r\n0,50,50\r\n1,10,10\r\n", 2, 100, 10);
    std::printf("malformed_csv_rejection=%d\n", pass);
    return pass;
}

bool run_case(const std::filesystem::path& root, double side,
              std::uint64_t expected_table_hash) {
    std::vector<float> x, y, x_repeat, y_repeat, x_other, y_other;
    pf::PalmieriInitDiagnostics d{};
    pf::palmieri_sequential_centres(72, side, 49.0, 1729, &x, &y, &d);
    pf::palmieri_sequential_centres(72, side, 49.0, 1729,
                                    &x_repeat, &y_repeat, nullptr);
    pf::palmieri_sequential_centres(72, side, 49.0, 1730,
                                    &x_other, &y_other, nullptr);
    const bool centre = x[0] == (float)(side / 2.0) &&
                        y[0] == (float)(side / 2.0);
    const bool deterministic = x == x_repeat && y == y_repeat;
    const bool different_seed = x != x_other || y != y_other;
    const bool separated = d.minimum_periodic_distance >= 49.0;
    const std::uint64_t table_hash =
        pf::palmieri_centre_table_fnv1a64(x, y);
    const bool golden_table = table_hash == expected_table_hash;

    const std::filesystem::path csv = root /
        (side == 777.0 ? "centres_777.csv" : "centres_800.csv");
    std::string error;
    bool roundtrip = pf::palmieri_write_centres_csv(csv.string(), x, y, &error);
    std::vector<float> loaded_x, loaded_y;
    pf::PalmieriCentresCsvDiagnostics loaded{};
    roundtrip = roundtrip && pf::palmieri_read_centres_csv(
        csv.string(), 72, side, 49.0, &loaded_x, &loaded_y, &loaded, &error);
    roundtrip = roundtrip && loaded_x == x && loaded_y == y &&
                loaded.table_fnv1a64 == table_hash;

    std::printf("L=%.0f accepted=72 candidates=%llu rejected=%llu "
                "min_distance=%.9g table_fnv1a64=%016llx centre=%d "
                "deterministic=%d different_seed=%d golden_table=%d "
                "csv_roundtrip=%d\n",
                side, (unsigned long long)d.candidates_drawn,
                (unsigned long long)d.candidates_rejected,
                d.minimum_periodic_distance, (unsigned long long)table_hash,
                centre, deterministic, different_seed, golden_table, roundtrip);
    return centre && deterministic && different_seed && separated &&
           golden_table && roundtrip;
}

bool maximum_density_case() {
    std::vector<float> x, y;
    pf::PalmieriInitDiagnostics generated{};
    pf::palmieri_sequential_centres(1600, 3662.0, 49.0, 1729,
                                    &x, &y, &generated);
    pf::PalmieriCentresCsvDiagnostics checked{};
    std::string error;
    const bool pass = pf::palmieri_validate_centres(
        x, y, 1600, 3662.0, 49.0, &checked, &error) &&
        generated.candidates_drawn - generated.candidates_rejected == 1599 &&
        checked.accepted_count == 1600;
    std::printf("maximum_density_N1600=%d candidates=%llu rejected=%llu "
                "min_distance=%.9g\n", pass,
                (unsigned long long)generated.candidates_drawn,
                (unsigned long long)generated.candidates_rejected,
                checked.minimum_periodic_distance);
    return pass;
}

}  // namespace

int main() {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch().count();
    const std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("palmieri_initializer_test_" + std::to_string(nonce));
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        std::fprintf(stderr, "cannot create test directory: %s\n",
                     ec.message().c_str());
        return 2;
    }
    const bool u = uniform53_golden_vectors();
    const bool a = run_case(root, 777.0, 0x5702b22b79d1d05bull);
    const bool b = run_case(root, 800.0, 0x15ff80e8a59ac11eull);
    const bool malformed = malformed_cases(root);
    const bool scale = maximum_density_case();
    std::filesystem::remove_all(root, ec);
    const bool pass = u && a && b && malformed && scale;
    std::puts(pass ? "PALMIERI_INITIALIZER_TEST_PASS"
                   : "PALMIERI_INITIALIZER_TEST_FAIL");
    return pass ? 0 : 1;
}
