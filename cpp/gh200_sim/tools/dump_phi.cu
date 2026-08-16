// ===========================================================================
// dump_phi -- convert a --dump-state binary into numpy .npy arrays plus a CSV
// of per-cell observables, for comparison against the verified CPU oracle at
//   cpp/simulation/tests/python/cpu_reference.py
//
// The oracle keeps one full periodic (Ny, Nx) field per cell, which is far too
// large to ship for N=288 at L=1563. So the dump carries each cell's rect plus
// its global origin, and the Python side paints it into a zero-filled domain
// exactly as cpu_reference.cells_from_checkpoint does:
//
//   full = np.zeros((L, L))
//   ys = (gy0 + np.arange(wy)) % L
//   xs = (gx0 + np.arange(wx)) % L
//   full[np.ix_(ys, xs)] = phi_cell            # rect row-major, x fastest
//
// usage: dump_phi <state.bin> <outdir> [--composite]
// ===========================================================================

#include "../include/kernels.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace pf;

// ---------------------------------------------------------------------------
// Minimal .npy v1.0 writer. Header length must make (10 + len) a multiple of
// 64 and the header must end in '\n'.
// ---------------------------------------------------------------------------
static bool write_npy(const std::string& path, const void* data,
                      size_t elem_bytes, const char* descr,
                      const std::vector<size_t>& shape)
{
    std::string dict = "{'descr': '";
    dict += descr;
    dict += "', 'fortran_order': False, 'shape': (";
    size_t count = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        dict += std::to_string(shape[i]);
        dict += ",";
        count *= shape[i];
    }
    dict += "), }";

    size_t total = 10 + dict.size() + 1;            // + trailing '\n'
    const size_t pad = (64 - (total % 64)) % 64;
    dict.append(pad, ' ');
    dict += '\n';
    const uint16_t hlen = (uint16_t)dict.size();

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[error] cannot write %s\n", path.c_str());
        return false;
    }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    std::fwrite(magic, 1, 8, f);
    std::fwrite(&hlen, sizeof(hlen), 1, f);         // little-endian host (aarch64)
    std::fwrite(dict.data(), 1, dict.size(), f);
    if (count && data) std::fwrite(data, elem_bytes, count, f);
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <state.bin> <outdir> [--composite]\n"
            "  --composite  also write sum_i phi_i^2 over the whole domain\n"
            "               as composite_phi_sq.npy (float64, L x L)\n", argv[0]);
        return 2;
    }
    const std::string in_path = argv[1];
    const std::string out_dir = argv[2];
    bool composite = false;
    for (int i = 3; i < argc; ++i)
        if (!std::strcmp(argv[i], "--composite")) composite = true;

    std::FILE* f = std::fopen(in_path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[error] cannot open %s\n", in_path.c_str());
        return 3;
    }

    DumpHeader h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1) {
        std::fprintf(stderr, "[error] short read on the header\n");
        std::fclose(f);
        return 3;
    }
    if (h.magic != kDumpMagic) {
        std::fprintf(stderr, "[error] bad magic 0x%08x (expected 0x%08x)\n",
                     h.magic, kDumpMagic);
        std::fclose(f);
        return 3;
    }
    if (h.version != kDumpVersion) {
        std::fprintf(stderr, "[error] dump version %u, this tool understands %u\n",
                     h.version, kDumpVersion);
        std::fclose(f);
        return 3;
    }

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::fprintf(stderr, "[error] cannot create %s: %s\n", out_dir.c_str(),
                     ec.message().c_str());
        std::fclose(f);
        return 3;
    }

    const int N = h.num_cells;
    const int L = h.domain_side;
    std::printf("dump: %d cells, domain %d x %d, step %lld, t = %.6f\n",
                N, L, L, (long long)h.step, h.t_now);

    std::vector<int32_t> origins((size_t)N * 2), classes((size_t)N);
    std::vector<double>  composite_field;
    if (composite) composite_field.assign((size_t)L * (size_t)L, 0.0);

    const std::string csv_path = out_dir + "/cells.csv";
    std::FILE* csv = std::fopen(csv_path.c_str(), "w");
    if (!csv) {
        std::fprintf(stderr, "[error] cannot write %s\n", csv_path.c_str());
        std::fclose(f);
        return 3;
    }
    std::fprintf(csv, "id,cls,gx0,gy0,wx,wy,bb_lo_x,bb_hi_x,bb_lo_y,bb_hi_y,"
                      "gamma,v_A,theta,vx,vy,phi_max,V,Cx,Cy,perim,Ix,Iy,"
                      "com_x,com_y\n");

    std::vector<float> phi;
    bool uniform_class = true;
    int cls0 = -1;
    std::vector<float> all;

    for (int n = 0; n < N; ++n) {
        DumpCell c{};
        if (std::fread(&c, sizeof(c), 1, f) != 1) {
            std::fprintf(stderr, "[error] short read on cell %d\n", n);
            std::fclose(csv); std::fclose(f); return 3;
        }
        if (c.wx <= 0 || c.wy <= 0 || (long long)c.wx * c.wy > 1 << 22) {
            std::fprintf(stderr, "[error] implausible rect %dx%d on cell %d\n",
                         c.wx, c.wy, n);
            std::fclose(csv); std::fclose(f); return 3;
        }
        phi.assign((size_t)c.wx * (size_t)c.wy, 0.0f);
        if (std::fread(phi.data(), sizeof(float), phi.size(), f) != phi.size()) {
            std::fprintf(stderr, "[error] short read on phi of cell %d\n", n);
            std::fclose(csv); std::fclose(f); return 3;
        }

        if (cls0 < 0) cls0 = c.cls;
        else if (c.cls != cls0) uniform_class = false;

        origins[(size_t)n * 2 + 0] = c.gx0;
        origins[(size_t)n * 2 + 1] = c.gy0;
        classes[(size_t)n] = c.cls;

        const double V = c.V > 0.0 ? c.V : 1.0;
        const double comx = (double)c.gx0 + c.Cx / V;
        const double comy = (double)c.gy0 + c.Cy / V;

        std::fprintf(csv,
            "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
            "%.9g,%.9g,%.9g,%.17g,%.17g,%.9g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.9f,%.9f\n",
            c.global_id, c.cls, c.gx0, c.gy0, c.wx, c.wy,
            c.bb_lo_x, c.bb_hi_x, c.bb_lo_y, c.bb_hi_y,
            (double)c.gamma, (double)c.v_A, (double)c.theta,
            (double)c.vx, (double)c.vy, (double)c.phi_max,
            c.V, c.Cx, c.Cy, c.perim, c.Ix, c.Iy,
            comx - std::floor(comx / (double)L) * (double)L,
            comy - std::floor(comy / (double)L) * (double)L);

        char name[256];
        std::snprintf(name, sizeof(name), "%s/phi_%05d.npy", out_dir.c_str(), n);
        if (!write_npy(name, phi.data(), sizeof(float), "<f4",
                       {(size_t)c.wy, (size_t)c.wx})) {
            std::fclose(csv); std::fclose(f); return 3;
        }
        if (uniform_class) all.insert(all.end(), phi.begin(), phi.end());

        if (composite) {
            for (int b = 0; b < c.wy; ++b) {
                const int gy = ((c.gy0 + b) % L + L) % L;
                for (int a = 0; a < c.wx; ++a) {
                    const int gx = ((c.gx0 + a) % L + L) % L;
                    const double v = (double)phi[(size_t)b * c.wx + a];
                    composite_field[(size_t)gy * L + gx] += v * v;
                }
            }
        }
    }
    std::fclose(csv);
    std::fclose(f);

    write_npy(out_dir + "/origins.npy", origins.data(), sizeof(int32_t), "<i4",
              {(size_t)N, 2});
    write_npy(out_dir + "/classes.npy", classes.data(), sizeof(int32_t), "<i4",
              {(size_t)N});
    if (uniform_class && cls0 >= 0 && cls0 < h.num_classes) {
        write_npy(out_dir + "/phi_all.npy", all.data(), sizeof(float), "<f4",
                  {(size_t)N, (size_t)h.cls_wy[cls0], (size_t)h.cls_wx[cls0]});
    }
    if (composite) {
        write_npy(out_dir + "/composite_phi_sq.npy", composite_field.data(),
                  sizeof(double), "<f8", {(size_t)L, (size_t)L});
    }

    const std::string meta = out_dir + "/meta.json";
    std::FILE* mf = std::fopen(meta.c_str(), "w");
    if (mf) {
        std::fprintf(mf,
            "{\n"
            "  \"num_cells\": %d,\n"
            "  \"Nx\": %d, \"Ny\": %d,\n"
            "  \"tile_pitch\": %d,\n"
            "  \"step\": %lld, \"t\": %.17g,\n"
            "  \"dx\": %.17g, \"dy\": %.17g, \"dt\": %.17g,\n"
            "  \"lambda\": %.17g, \"target_radius\": %.17g,\n"
            "  \"kappa\": %.17g, \"mu\": %.17g, \"xi\": %.17g,\n"
            "  \"tau\": %.17g, \"v_A\": %.17g,\n"
            "  \"gamma\": %.17g, \"gamma_cancer\": %.17g,\n"
            "  \"p_tumble\": %.17g,\n"
            "  \"uniform_class\": %s,\n"
            "  \"flags\": [%u, %u, %u, %u, %u, %u, %u, %u]\n"
            "}\n",
            N, L, L, h.tile_pitch, (long long)h.step, h.t_now,
            h.dx, h.dy, h.dt, h.lambda, h.radius, h.kappa, h.mu, h.xi,
            h.tau, h.v_A, h.gamma_normal, h.gamma_cancer, h.p_tumble,
            uniform_class ? "true" : "false",
            h.flags[0], h.flags[1], h.flags[2], h.flags[3],
            h.flags[4], h.flags[5], h.flags[6], h.flags[7]);
        std::fclose(mf);
    }

    std::printf("wrote %s/{meta.json,cells.csv,origins.npy,classes.npy,"
                "phi_%%05d.npy%s%s}\n", out_dir.c_str(),
                uniform_class ? ",phi_all.npy" : "",
                composite ? ",composite_phi_sq.npy" : "");
    for (int i = 0; i < FLAG_COUNT; ++i)
        if (h.flags[i])
            std::fprintf(stderr, "[alarm] %s = %u -- the source run is INVALID\n",
                         flag_name(i), h.flags[i]);
    return 0;
}
