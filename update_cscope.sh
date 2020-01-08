#!/bin/bash

find . \( -name "*.c" -or \
-name "*.cpp" -or \
-name "*.cc" -or \
-name "*.h" -or \
-name "*.hpp" -or \
-name "*.output" -or \
-name "*.inc" \) \
-and -not -path "./build/*" \
-and -not -path "./build_*/*" > cscope.files

cscope -bq
