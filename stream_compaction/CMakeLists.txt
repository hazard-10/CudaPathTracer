set(SOURCE_FILES
	common.h
	common.cu
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    )