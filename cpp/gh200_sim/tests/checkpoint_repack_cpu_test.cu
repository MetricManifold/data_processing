// Compile checkpoint.cu into this test translation unit so the CPU test calls
// the production repack_tile implementation, not a duplicate. The selected
// path performs no CUDA API call; nvcc is used only as the host compiler/parser.
#include "../src/checkpoint.cu"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kSourceTile = 256;

std::vector<float> make_source(int support_x) {
    std::vector<float> source((size_t)kSourceTile * kSourceTile, 0.0f);
    const int support_x0 = support_x == 200 ? 36 : 32;
    const int support_x1 = support_x0 + support_x - 1;
    for (int y = 80; y <= 192; ++y)
        for (int x = support_x0; x <= support_x1; ++x)
            source[(size_t)y * kSourceTile + x] = 0.5f;

    // Preserve sub-threshold tails at the compact terminal window's corners.
    source[(size_t)32 * kSourceTile + 32] = 1.0e-7f;
    source[(size_t)239 * kSourceTile + 239] = -1.0e-7f;
    return source;
}

bool verify_exact_copy(const std::vector<float>& source,
                       const std::vector<float>& destination) {
    for (int y = 0; y < pf::kTilePitch; ++y) {
        for (int x = 0; x < pf::kTilePitch; ++x) {
            const float expected =
                x < kSourceTile && y < kSourceTile
                    ? source[(size_t)y * kSourceTile + x]
                    : 0.0f;
            if (destination[(size_t)y * pf::kTilePitch + x] != expected)
                return false;
        }
    }
    return true;
}

bool repack_case(int support_x, bool should_succeed) {
    const std::vector<float> source = make_source(support_x);
    std::vector<float> destination((size_t)pf::kTileArea, -9.0f);
    int cls = -1;
    int offset[2] = {-1, -1};
    int extent[2] = {-1, -1};
    float dropped = -1.0f;
    bool exact = false;
    const bool ok = pf::repack_tile(
        source.data(), kSourceTile, 55, destination.data(), &cls, offset,
        &dropped, &exact, extent);
    if (ok != should_succeed) return false;
    if (!ok) return true;
    const int expected_cls = support_x <= pf::kLargeClassEdge
                           ? pf::kClassLarge : pf::kClassFallback;
    const pf::ShapeClass expected = pf::class_of(expected_cls);
    return exact && cls == expected_cls && offset[0] == expected.tx0 &&
           offset[1] == expected.ty0 && extent[0] == support_x && extent[1] == 113 &&
           dropped == 0.0f && verify_exact_copy(source, destination);
}

bool failed_seed_case() {
    constexpr int T = 288;
    std::vector<float> source((size_t)T * T, 0.0f);
    for (int y = 34; y <= 250; ++y)
        for (int x = 86; x <= 219; ++x)
            source[(size_t)y * T + x] = 0.5f;
    source[(size_t)32 * T + 32] = 1.0e-7f;
    source[(size_t)255 * T + 255] = -1.0e-7f;

    std::vector<float> destination((size_t)pf::kTileArea, -9.0f);
    int cls = -1, offset[2] = {-1, -1}, extent[2] = {-1, -1};
    float dropped = -1.0f;
    bool exact = false;
    const bool ok = pf::repack_tile(
        source.data(), T, 0, destination.data(), &cls, offset,
        &dropped, &exact, extent);
    if (!ok || !exact || cls != pf::kClassFallback ||
        offset[0] != 1 || offset[1] != 1 ||
        extent[0] != 134 || extent[1] != 217 || dropped != 0.0f)
        return false;
    return destination == source;
}

}  // namespace

int main() {
    if (!repack_case(200, true)) return 1;
    if (!repack_case(208, true)) return 2;
    if (pf::kExtendedSupportLayout && !failed_seed_case()) return 3;
    std::printf("layout=%s\n",
                pf::kExtendedSupportLayout ? "extended" : "compact");
    std::printf("file_tile=256\n");
    std::printf("native_tile=%d\n", pf::kTilePitch);
    std::printf("support200_exact=1\n");
    std::printf("support208_accepted=1\n");
    std::printf("failed_seed_134x217_exact=%d\n",
                pf::kExtendedSupportLayout ? 1 : 0);
    std::puts("CHECKPOINT_REPACK_CPU_TEST_PASS");
    return 0;
}
