#!/bin/bash
export LANG=
set -e
testname=$(basename -s .sh "$0")
echo -n "Testing $testname ... "
cd "$(dirname "$0")"/../..
mold="$(pwd)/mold"
t="$(pwd)/out/test/elf/$testname"
mkdir -p "$t"

echo '.globl main; main:' | cc -o "$t"/a.o -c -x assembler -

clang -fuse-ld="$mold" -o "$t"/exe "$t"/a.o

readelf --dynamic "$t"/exe > "$t"/log
grep -Pq 'Shared library:.*\blibc.so\b' "$t"/log

readelf -W --dyn-syms --use-dynamic "$t"/exe > "$t"/log2
grep -Pq 'FUNC\s+GLOBAL\s+DEFAULT\s+UND\s+__libc_start_main' "$t"/log2

cat <<EOF | clang -c -fPIC -o "$t"/b.o -xc -
#include <stdio.h>

int main() {
  printf("Hello world\n");
}
EOF

clang -fuse-ld="$mold" -o "$t"/exe -pie "$t"/b.o
count=$(readelf -W --relocs "$t"/exe | grep -P 'R_[\w\d_]+_RELATIVE' | wc -l)
readelf -W --dynamic "$t"/exe | grep -q "RELACOUNT.*\b$count\b"

echo OK
