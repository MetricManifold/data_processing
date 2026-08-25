# Historical filename retained so downstream include paths do not drift.  The
# old edge-only selector is unsafe for a 224-pixel class because tile=256 and
# class=224 cannot retain the aligned origin plus zero ring.  Select the two
# audited quantities atomically instead.
if(DEFINED PF_LARGE_CLASS_EDGE)
  message(FATAL_ERROR
    "PF_LARGE_CLASS_EDGE is retired: tile pitch and shared-phi edge must change "
    "as one audited pair. Use -DPF_EXTENDED_SUPPORT_LAYOUT=ON or omit both "
    "options for the compact production layout.")
endif()

option(PF_EXTENDED_SUPPORT_LAYOUT
       "Default ON: tile=288, shared class=224, global fallback interior=286"
       ON)

if(PF_EXTENDED_SUPPORT_LAYOUT)
  set(PF_EXTENDED_SUPPORT_LAYOUT_VALUE 1)
  set(PF_SUPPORT_TILE_PITCH 288)
  set(PF_SUPPORT_LARGE_EDGE 224)
  message(STATUS
    "gh200_sim: EXTENDED layout (tile 288, shared-phi class 224, "
    "global fallback 286 with 278 px guarded support capacity).")
else()
  set(PF_EXTENDED_SUPPORT_LAYOUT_VALUE 0)
  set(PF_SUPPORT_TILE_PITCH 256)
  set(PF_SUPPORT_LARGE_EDGE 208)
  message(WARNING
    "gh200_sim: COMPACT layout (tile 256, shared-phi class 208, "
    "global fallback 254 with 246 px guarded support capacity).")
endif()
