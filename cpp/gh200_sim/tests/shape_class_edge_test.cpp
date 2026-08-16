#include "kernels.cuh"
#include "checkpoint_format.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef PF_EXPECTED_TILE_PITCH
#error "PF_EXPECTED_TILE_PITCH is required by this test"
#endif
#ifndef PF_EXPECTED_LARGE_CLASS_EDGE
#error "PF_EXPECTED_LARGE_CLASS_EDGE is required by this test"
#endif

namespace {

using namespace pf;

static_assert(kTilePitch == PF_EXPECTED_TILE_PITCH,
              "test was compiled for the wrong tile layout");
static_assert(kLargeClassEdge == PF_EXPECTED_LARGE_CLASS_EDGE,
              "test was compiled for the wrong terminal edge");
static_assert((kTilePitch == 256 && kLargeClassEdge == 208) ||
              (kTilePitch == 288 && kLargeClassEdge == 224));

// The option may alter only the storage tile and terminal phi-only edge.
static_assert(kClasses[0].wx == 144 && kClasses[0].wy == 144 &&
              kClasses[0].tx0 == 64 && kClasses[0].ty0 == 64);
static_assert(kClasses[1].wx == 176 && kClasses[1].wy == 144 &&
              kClasses[1].tx0 == 32 && kClasses[1].ty0 == 64);
static_assert(kClasses[2].wx == 144 && kClasses[2].wy == 176 &&
              kClasses[2].tx0 == 64 && kClasses[2].ty0 == 32);
static_assert(kClasses[3].wx == 160 && kClasses[3].wy == 160 &&
              kClasses[3].tx0 == 32 && kClasses[3].ty0 == 32);
static_assert(kClasses[4].wx == kLargeClassEdge &&
              kClasses[4].wy == kLargeClassEdge &&
              kClasses[4].tx0 == 32 && kClasses[4].ty0 == 32);

static_assert(detail::class_ok(kClasses[kClassLarge]));
static_assert(detail::cpasync_dst_ok(kClasses[kClassLarge]));
static_assert(kTilePitch % 32 == 0);
static_assert(kTileArea == (kExtendedSupportLayout ? 82944 : 65536));
static_assert(kSmemRaw == 213440);
static_assert(kSmemBytes == 213504);
static_assert(kLargeClassSmemRaw ==
              (kExtendedSupportLayout ? 211904 : 183616));
static_assert(kLargeClassSmemBytes ==
              (kExtendedSupportLayout ? 211968 : 183680));
static_assert(kLargeClassMarginToStaged ==
              (kExtendedSupportLayout ? 1536 : 29824));
static_assert(kTilePitch -
                  (kClasses[kClassLarge].tx0 + kLargeClassEdge) ==
              (kExtendedSupportLayout ? 32 : 16));

static_assert(class_containing(kLargeClassEdge - kPromoteSlack,
                               kLargeClassEdge - kPromoteSlack,
                               kPromoteSlack) == kClassLarge);
static_assert(class_containing(kLargeClassEdge - kPromoteSlack + 1,
                               kLargeClassEdge - kPromoteSlack + 1,
                               kPromoteSlack) == -1);
static_assert(class_containing(200, 113, kPromoteSlack) == kClassLarge);
static_assert(class_containing(201, 113, kPromoteSlack) ==
              (kExtendedSupportLayout ? kClassLarge : -1));
static_assert(class_containing(208, 113, kPromoteSlack) ==
              (kExtendedSupportLayout ? kClassLarge : -1));

// The compact checkpoint's terminal window is source coordinates 32..239.
// Both layouts preserve a 200-pixel boundary state exactly; only the extended
// candidate can admit all possible 201..208-pixel first-trigger extents.
static_assert(class_preserving_nonzero(
                  200, 113, kPromoteSlack, 32, 239, 32, 239) == kClassLarge);
static_assert(class_preserving_nonzero(
                  201, 113, kPromoteSlack, 32, 239, 32, 239) ==
              (kExtendedSupportLayout ? kClassLarge : -1));
static_assert(class_preserving_nonzero(
                  208, 113, kPromoteSlack, 32, 239, 32, 239) ==
              (kExtendedSupportLayout ? kClassLarge : -1));

// tile_t is data in v8; changing its value does not change the schema.
static_assert(ckpt::VERSION_CURRENT == 8);
static_assert(sizeof(ckpt::FixedPrefix) == 44);
static_assert(sizeof(ckpt::RankTrailer) == 12);
static_assert(sizeof(ckpt::CellRecordHeader) == 32);
static_assert(sizeof(ckpt::SimParamsV8) == 144);

uint64_t bits_of(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint32_t bits_of(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void print_class(int index) {
    const ShapeClass c = kClasses[index];
    std::printf("class%d=%d,%d,%d,%d\n", index, c.wx, c.wy, c.tx0, c.ty0);
}

}  // namespace

int main() {
    pf::SimParams p{};
    p.dt = 0.01;
    p.tau = 10000.0;
    p.seed = 800001ull;
    p.polarity_seed = 80090001ull;
    p.kappa = 10.0;
    p.xi = 1500.0;
    p.lambda = 7.0;
    p.mu = 1.0;

    const auto stream = p.polarity_stream();
    const pf::Philox4 r = pf::philox4x32_10(
        140650001u, 0u, 55u, 0xA5A5A5A5u,
        static_cast<uint32_t>(stream), static_cast<uint32_t>(stream >> 32));
    const double u = pf::philox_uniform53(r.v[0], r.v[1]);
    const float theta = pf::ic_theta(55, stream);

    std::printf("layout=%s\n", pf::kExtendedSupportLayout
                                   ? "extended" : "compact");
    std::printf("tile_pitch=%d\n", pf::kTilePitch);
    std::printf("tile_area=%d\n", pf::kTileArea);
    std::printf("large_edge=%d\n", pf::kLargeClassEdge);
    for (int c = 0; c < pf::kNumClasses; ++c) print_class(c);
    std::printf("large_raw=%d\n", pf::kLargeClassSmemRaw);
    std::printf("large_aligned=%d\n", pf::kLargeClassSmemBytes);
    std::printf("staged_raw=%d\n", pf::kSmemRaw);
    std::printf("launch_aligned=%d\n", pf::kSmemBytes);
    std::printf("large_to_staged_margin=%d\n",
                pf::kLargeClassMarginToStaged);
    std::printf("tile_right_margin=%d\n",
                pf::kTilePitch -
                    (pf::kClasses[pf::kClassLarge].tx0 + pf::kLargeClassEdge));
    std::printf("max_support_extent=%d\n",
                pf::kLargeClassEdge - pf::kPromoteSlack);
    std::printf("extent201_class=%d\n",
                pf::class_containing(201, 113, pf::kPromoteSlack));
    std::printf("extent208_class=%d\n",
                pf::class_containing(208, 113, pf::kPromoteSlack));
    std::printf("old_tile_exact_200=%d\n",
                pf::class_preserving_nonzero(
                    200, 113, pf::kPromoteSlack, 32, 239, 32, 239));
    std::printf("old_tile_exact_208=%d\n",
                pf::class_preserving_nonzero(
                    208, 113, pf::kPromoteSlack, 32, 239, 32, 239));

    // These values must be byte-identical across both compilations.
    std::printf("p_tumble_bits=%016llx\n",
                static_cast<unsigned long long>(bits_of(p.p_tumble())));
    std::printf("polarity_stream=%llu\n",
                static_cast<unsigned long long>(stream));
    std::printf("philox=%08x,%08x,%08x,%08x\n",
                r.v[0], r.v[1], r.v[2], r.v[3]);
    std::printf("uniform_bits=%016llx\n",
                static_cast<unsigned long long>(bits_of(u)));
    std::printf("theta_bits=%08x\n", bits_of(theta));
    std::printf("checkpoint_version=%u\n", ckpt::VERSION_CURRENT);
    std::printf("checkpoint_sizes=%zu,%zu,%zu,%zu\n",
                sizeof(ckpt::FixedPrefix), sizeof(ckpt::RankTrailer),
                sizeof(ckpt::CellRecordHeader), sizeof(ckpt::SimParamsV8));
    std::puts("SUPPORT_LAYOUT_TEST_PASS");
    return 0;
}
