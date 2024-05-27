# This CMake script copies binary SPIR-V data into a .cpp file.
# It is meant to be used as a standalone CMake file invoked at build-time.
file(READ ${INPUT_FILE} FILE_CONTENTS HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])" "0x\\4\\3\\2\\1U," FILE_CONTENTS ${FILE_CONTENTS})
configure_file(${CMAKE_CURRENT_LIST_DIR}/spirv_embed.cpp.in ${OUTPUT_FILE})
