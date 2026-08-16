# Historical filename retained so downstream include paths do not drift.  The
# old edge-only selector is unsafe for a 224-pixel class because tile=256 and
# class=224 cannot retain the aligned origin plus zero ring.  Select the two
# audited quantities atomically instead.
if(DEFINED PF_LARGE_CLASS_EDGE)
  message(FATAL_ERROR
    "PF_LARGE_CLASS_EDGE is retired: tile pitch and terminal edge must change "
    "as one audited pair. Use -DPF_EXTENDED_SUPPORT_LAYOUT=ON or omit both "
    "options for the compact production layout.")
endif()

option(PF_EXTENDED_SUPPORT_LAYOUT
       "Default ON: tile=288 and terminal class=224 (216 px support capacity)"
       ON)

if(PF_EXTENDED_SUPPORT_LAYOUT)
  set(PF_EXTENDED_SUPPORT_LAYOUT_VALUE 1)
  set(PF_SUPPORT_TILE_PITCH 288)
  set(PF_SUPPORT_LARGE_EDGE 224)
  message(STATUS
    "gh200_sim: EXTENDED support layout, default (tile 288, class 224, "
    "216 px/axis capacity). GPU-gated on Roihu job 687115 for support "
    "capacity and 1/10/100-step restart parity; a full-length production "
    "segment is still unmeasured.")
else()
  set(PF_EXTENDED_SUPPORT_LAYOUT_VALUE 0)
  set(PF_SUPPORT_TILE_PITCH 256)
  set(PF_SUPPORT_LARGE_EDGE 208)
  message(WARNING
    "gh200_sim: COMPACT LEGACY support layout (tile 256, class 208, "
    "200 px/axis capacity). This is the geometry whose capacity the N=800 "
    "soft branch exhausted (job 666491). Select it only to reproduce a "
    "pre-2026-08-16 run.")
endif()
