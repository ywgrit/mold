// LoongArch is a new RISC ISA announced in 2021 by Loongson. The ISA
// feels like a modernized MIPS with a hint of RISC-V flavor, although
// it's not compatible with either one.
//
// While LoongArch is a fresh and clean ISA, its technological advantage
// over other modern RISC ISAs such as RISC-V doesn't seem to be very
// significant. It appears that the real selling point of LoongArch is
// that the ISA is developed and controlled by a Chinese company,
// reflecting a desire for domestic CPUs. Loongson is actively working on
// bootstrapping the entire ecosystem for LoongArch, sending patches to
// Linux, GCC, LLVM, etc.
//
// All instructions are 4 bytes long in LoongArch and aligned to 4-byte
// boundaries. It has 32 general-purpose registers. Among these, $t0 - $t8
// (aliases for $r12 - $r20) are temporary registers that we can use in
// our PLT and range extension thunks.
//
// The psABI defines a few linker relaxations. We haven't supported them
// yet.
//
// https://loongson.github.io/LoongArch-Documentation/LoongArch-ELF-ABI-EN.html

#if MOLD_LOONGARCH64 || MOLD_LOONGARCH32

#include "mold.h"

#include <tbb/parallel_for_each.h>

namespace mold::elf {

using E = MOLD_TARGET;

#define LOONGARCH_MAX_PAGESIZE 16384

static u64 page(u64 val) {
  return val & 0xffff'ffff'ffff'f000;
}

static u64 hi20(u64 val, u64 pc) {
  // A PC-relative address with a 32 bit offset is materialized in a
  // register with the following instructions:
  //
  //   pcalau12i $rN, %pc_hi20(sym)
  //   addi.d    $rN, $rN, %lo12(sym)
  //
  // PCALAU12I materializes bits [63:12] by computing (pc + imm << 12)
  // and zero-clear [11:0]. ADDI.D sign-extends its 12 bit immediate and
  // add it to the register. To compensate the sign-extension, PCALAU12I
  // needs to materialize a 0x1000 larger value than the desired [63:12]
  // if [11:0] is sign-extended.
  //
  // This is similar but different from RISC-V because RISC-V's AUIPC
  // doesn't zero-clear [11:0].
  return bits(page(val + 0x800) - page(pc), 31, 12);
}

static u64 hi64(u64 val, u64 pc) {
  // A PC-relative 64-bit address is materialized with the following
  // instructions for the large code model:
  //
  //   pcalau12i $rN, %pc_hi20(sym)
  //   addi.d    $rM, $zero, %lo12(sym)
  //   lu32i.d   $rM, %pc64_lo20(sym)
  //   lu52i.d   $rM, $r12, %pc64_hi12(sym)
  //   add.d     $rN, $rN, $rM
  //
  // PCALAU12I computes (pc + imm << 12) to materialize a 64-bit value.
  // ADDI.D adds a sign-extended 12 bit value to a register. LU32I.D and
  // LU52I.D simply set bits to [51:31] and to [63:53], respectively.
  //
  // Compensating all the sign-extensions is a bit complicated.
  u64 x = page(val) - page(pc);
  if (val & 0x800)
    x += 0x1000 - 0x1'0000'0000;
  if (x & 0x8000'0000)
    x += 0x1'0000'0000;
  return x;
}

static u64 higher20(u64 val, u64 pc) {
  return bits(hi64(val, pc), 51, 32);
}

static u64 highest12(u64 val, u64 pc) {
  return bits(hi64(val, pc), 63, 52);
}

static void write_k12(u8 *loc, u32 val) {
  // opcode, [11:0], rj, rd
  *(ul32 *)loc &= 0b1111111111'000000000000'11111'11111;
  *(ul32 *)loc |= bits(val, 11, 0) << 10;
}

static void write_k16(u8 *loc, u32 val) {
  // opcode, [15:0], rj, rd
  *(ul32 *)loc &= 0b111111'0000000000000000'11111'11111;
  *(ul32 *)loc |= bits(val, 15, 0) << 10;
}

static void write_j20(u8 *loc, u32 val) {
  // opcode, [19:0], rd
  *(ul32 *)loc &= 0b1111111'00000000000000000000'11111;
  *(ul32 *)loc |= bits(val, 19, 0) << 5;
}

static void write_d5k16(u8 *loc, u32 val) {
  // opcode, [15:0], rj, [20:16]
  *(ul32 *)loc &= 0b111111'0000000000000000'11111'00000;
  *(ul32 *)loc |= bits(val, 15, 0) << 10;
  *(ul32 *)loc |= bits(val, 20, 16);
}

static void write_d10k16(u8 *loc, u32 val) {
  // opcode, [15:0], [25:16]
  *(ul32 *)loc &= 0b111111'0000000000000000'0000000000;
  *(ul32 *)loc |= bits(val, 15, 0) << 10;
  *(ul32 *)loc |= bits(val, 25, 16);
}

template <>
void write_plt_header<E>(Context<E> &ctx, u8 *buf) {
  static const ul32 insn_64[] = {
    0x1a00'000e, // pcalau12i $t2, %pc_hi20(.got.plt)
    0x0011'bdad, // sub.d     $t1, $t1, $t3
    0x28c0'01cf, // ld.d      $t3, $t2, %lo12(.got.plt) # _dl_runtime_resolve
    0x02ff'51ad, // addi.d    $t1, $t1, -44             # .plt entry
    0x02c0'01cc, // addi.d    $t0, $t2, %lo12(.got.plt) # &.got.plt
    0x0045'05ad, // srli.d    $t1, $t1, 1               # .plt entry offset
    0x28c0'218c, // ld.d      $t0, $t0, 8               # link map
    0x4c00'01e0, // jr        $t3
  };

  static const ul32 insn_32[] = {
    0x1a00'000e, // pcalau12i $t2, %pc_hi20(.got.plt)
    0x0011'3dad, // sub.w     $t1, $t1, $t3
    0x2880'01cf, // ld.w      $t3, $t2, %lo12(.got.plt) # _dl_runtime_resolve
    0x02bf'51ad, // addi.w    $t1, $t1, -44             # .plt entry
    0x0280'01cc, // addi.w    $t0, $t2, %lo12(.got.plt) # &.got.plt
    0x0044'89ad, // srli.w    $t1, $t1, 2               # .plt entry offset
    0x2880'118c, // ld.w      $t0, $t0, 4               # link map
    0x4c00'01e0, // jr        $t3
  };

  u64 gotplt = ctx.gotplt->shdr.sh_addr;
  u64 plt = ctx.plt->shdr.sh_addr;

  memcpy(buf, E::is_64 ? insn_64 : insn_32, E::plt_hdr_size);
  write_j20(buf, hi20(gotplt, plt));
  write_k12(buf + 8, gotplt);
  write_k12(buf + 16, gotplt);
}

static const ul32 plt_entry_64[] = {
  0x1a00'000f, // pcalau12i $t3, %pc_hi20(func@.got.plt)
  0x28c0'01ef, // ld.d      $t3, $t3, %lo12(func@.got.plt)
  0x4c00'01ed, // jirl      $t1, $t3, 0
  0x0340'0000, // nop
};

static const ul32 plt_entry_32[] = {
  0x1a00'000f, // pcalau12i $t3, %pc_hi20(func@.got.plt)
  0x2880'01ef, // ld.w      $t3, $t3, %lo12(func@.got.plt)
  0x4c00'01ed, // jirl      $t1, $t3, 0
  0x0340'0000, // nop
};

template <>
void write_plt_entry<E>(Context<E> &ctx, u8 *buf, Symbol<E> &sym) {
  u64 gotplt = sym.get_gotplt_addr(ctx);
  u64 plt = sym.get_plt_addr(ctx);

  memcpy(buf, E::is_64 ? plt_entry_64 : plt_entry_32, E::plt_size);
  write_j20(buf, hi20(gotplt, plt));
  write_k12(buf + 4, gotplt);
}

template <>
void write_pltgot_entry<E>(Context<E> &ctx, u8 *buf, Symbol<E> &sym) {
  u64 got = sym.get_got_pltgot_addr(ctx);
  u64 plt = sym.get_plt_addr(ctx);

  memcpy(buf, E::is_64 ? plt_entry_64 : plt_entry_32, E::plt_size);
  write_j20(buf, hi20(got, plt));
  write_k12(buf + 4, got);
}

template <>
void EhFrameSection<E>::apply_eh_reloc(Context<E> &ctx, const ElfRel<E> &rel,
                                       u64 offset, u64 val) {
  u8 *loc = ctx.buf + this->shdr.sh_offset + offset;

  switch (rel.r_type) {
  case R_NONE:
    break;
  case R_LARCH_ADD6:
    *loc = (*loc & 0b1100'0000) | ((*loc + val) & 0b0011'1111);
    break;
  case R_LARCH_ADD8:
    *loc += val;
    break;
  case R_LARCH_ADD16:
    *(ul16 *)loc += val;
    break;
  case R_LARCH_ADD32:
    *(ul32 *)loc += val;
    break;
  case R_LARCH_ADD64:
    *(ul64 *)loc += val;
    break;
  case R_LARCH_SUB6:
    *loc = (*loc & 0b1100'0000) | ((*loc - val) & 0b0011'1111);
    break;
  case R_LARCH_SUB8:
    *loc -= val;
    break;
  case R_LARCH_SUB16:
    *(ul16 *)loc -= val;
    break;
  case R_LARCH_SUB32:
    *(ul32 *)loc -= val;
    break;
  case R_LARCH_SUB64:
    *(ul64 *)loc -= val;
    break;
  case R_LARCH_32_PCREL:
    *(ul32 *)loc = val - this->shdr.sh_addr - offset;
    break;
  case R_LARCH_64_PCREL:
    *(ul64 *)loc = val - this->shdr.sh_addr - offset;
    break;
  default:
    Fatal(ctx) << "unsupported relocation in .eh_frame: " << rel;
  }
}

static void
putl32(u32 val, u8 *loc) {
  loc[0] = val & 0xff;
  loc[1] = (val >> 8) & 0xff;
  loc[2] = (val >> 16) & 0xff;
  loc[3] = (val >> 24) & 0xff;
}

// when delete bytes, we need to update both InputSection and OutputSection
template <typename E>
static bool
loongarch_relax_delete_bytes(Context<E> &ctx, InputSection<E> *sec, u64 addr, u64 count, u8 *contents) { // content is in outputsection
    u64 i;
    u64 sec_shndx = sec->shndx;
    u64 toaddr = sec->sh_size;
    ElfShdr<E> *shdr;
    if (sec->shndx < sec->file.elf_sections.size())
	    shdr = &(sec->file.elf_sections[sec->shndx]);
    else
	    shdr = &(sec->file.elf_sections2[sec->shndx - sec->file.elf_sections.size()]);
    shdr->sh_size -= count; // NB: this will not reduce the size of sections of outputfile because the sections of outputfile have been set before apply_reloc_[no]_alloc
    sec->sh_size -= count;
    memmove (contents + addr, contents + addr + count, toaddr - addr - count); // the last count bytes will be clear in ref:OutputSection<E>::write_to

    std::span<ElfRel<E>> rels = sec->get_rels(ctx);
    for (i = 0; i < rels.size(); i++)
    {
        if (rels[i].r_offset > addr && rels[i].r_offset < toaddr)
            rels[i].r_offset -= count;
    }

  /* Adjust the local symbols defined in this section. */
  std::vector<Symbol<E> *> &symbols = sec->file.symbols;
  for (i = 0; i < symbols.size(); i++)
    {
      Symbol<E> *sym = symbols[i];
      ElfSym<E> &esym = sec->file.elf_syms[i];
      if (esym.is_undef())
	      continue;

      if (esym.st_shndx == sec_shndx) // According to gdb, esym.st_value is always same with sym->value.
	{
	  /* If the symbol is in the range of memory we just moved, we
	     have to adjust its value. */
	   if (esym.st_value > addr && esym.st_value <= toaddr)
      {
    	  sym->value -= count;
          esym.st_value -= count;
      }

	  /* If the symbol *spans* the bytes we just deleted (i.e. its
	     *end* is in the moved bytes but its *start* isn't), then we
	     must adjust its size.

	     This test needs to use the original value of st_value, otherwise
	     we might accidentally decrease size when deleting bytes right
	     before the symbol.  But since deleted relocs can't span across
	     symbols, we can't have both a st_value and a st_size decrease,
	     so it is simpler to just use an else.  */
	 else if (esym.st_value <= addr
		   && esym.st_value + esym.st_size > addr
		   && esym.st_value + esym.st_size <= toaddr)
      {
          esym.st_size -= count;
      }
	}
    }
   
  return true;
}

template <class E>
static bool loongarch_relax_pcala_addi(Context<E> &ctx, InputSection<E> *sec, InputSection<E> *sym_sec, ElfRel<E> *rel_hi, u64 symval,
                                u64 pc, u8 *contents) { // content is the first byte of InputSection rel_hi belongs to in OutputFile. symval and pc is the relative value from the start address of OutputFile.
    ElfRel<E> *rel_lo = rel_hi + 2;
    u32 pca = *(u32 *)(contents + rel_hi->r_offset);
    u32 add = *(u32 *)(contents + rel_lo->r_offset);
    u32 rd = pca & 0x1f;
    u64 max_alignment = 0;
    u64 ori_pc = pc;
    for (int i = 0; i < ctx.osec_pool.size(); i++)
        max_alignment = (u64)(ctx.osec_pool[i]->shdr.sh_addralign) > max_alignment ? (u64)(ctx.osec_pool[i]->shdr.sh_addralign)
      						  : max_alignment;
    if (sym_sec->shdr().sh_flags & SHF_WRITE) // If sym_sec is not readonly, then sym_sec not belongs to the fragment which contains rel_sec.
      {
        max_alignment = LOONGARCH_MAX_PAGESIZE > max_alignment ? LOONGARCH_MAX_PAGESIZE
      						  : max_alignment;
        if (symval > pc)
      pc -= max_alignment;
        else if (symval < pc)
      pc += max_alignment;
      }
    else
      if (symval > pc)
        pc -= max_alignment;
      else if (symval < pc)
        pc += max_alignment;

    const uint32_t addi_d = 0x02c00000;
    const uint32_t pcaddi = 0x18000000;
  
    /* Is pcalau12i + addi.d insns?  */
    if (rel_lo->r_type != R_LARCH_PCALA_LO12
        || (rel_lo + 1)->r_type != R_LARCH_RELAX
        || (rel_hi + 1)->r_type != R_LARCH_RELAX
        || rel_hi->r_offset + 4 != rel_lo->r_offset
        || (add & addi_d) != addi_d
        /* Is pcalau12i $rd + addi.d $rd,$rd?  */
        || (add & 0x1f) != rd
        || ((add >> 5) & 0x1f) != rd
        /* Can be relaxed to pcaddi?  */
        || symval & 0x3 /* 4 bytes align.  */
        || (long)(symval - pc) < (long)(int32_t)0xffe00000
        || (long)(symval - pc) > (long)(int32_t)0x1ffffc)
      return false;
  
    pca = pcaddi | rd;
    //bits(page(val + 0x800) - page(pc), 31, 12)
    u32 imm = (((symval - ori_pc) >> 2) << 5) & 0x01ffffe0;
    pca = pca | imm;

    putl32(pca, contents + rel_hi->r_offset);
   //putl32(0x03400000, contents + rel_lo->r_offset);
  
    /* Adjust relocations.  */
    //rel_hi->r_type = R_LARCH_PCREL20_S2;
    rel_hi->r_type = R_LARCH_NONE;
    rel_lo->r_sym = 0;
    rel_lo->r_type = R_LARCH_NONE;
  
    loongarch_relax_delete_bytes (ctx, sec, rel_lo->r_offset, 4, contents);
  
    return true;
}

template <>
void InputSection<E>::apply_reloc_alloc(Context<E> &ctx, u8 *base) { // base is the start address of this InputSection in OutputFile. The contents of InputSection has been copied to OutputSection before this function been invoked, now we can modify the contents of OutputSection in this function.
  std::span<const ElfRel<E>> rels = get_rels(ctx);

  ElfRel<E> *dynrel = nullptr;
  if (ctx.reldyn)
    dynrel = (ElfRel<E> *)(ctx.buf + ctx.reldyn->shdr.sh_offset +
                           file.reldyn_offset + this->reldyn_offset);

  for (i64 i = 0; i < rels.size(); i++) {
    const ElfRel<E> &rel = rels[i];

    if (rel.r_type == R_NONE || rel.r_type == R_LARCH_RELAX ||
        rel.r_type == R_LARCH_MARK_LA || rel.r_type == R_LARCH_MARK_PCREL ||
        rel.r_type == R_LARCH_ALIGN)
      continue;

    Symbol<E> &sym = *file.symbols[rel.r_sym]; // rel.r_sym is the index of file.symbols
    u8 *loc = base + rel.r_offset; // the absolute address of this reloc, write to loc is write to outputfile

    auto check = [&](i64 val, i64 lo, i64 hi) {
      if (val < lo || hi <= val)
        Error(ctx) << *this << ": relocation " << rel << " against "
                   << sym << " out of range: " << val << " is not in ["
                   << lo << ", " << hi << ")";
    };

    auto check_branch = [&](i64 val, i64 lo, i64 hi) {
      if (val & 0b11)
        Error(ctx) << *this << ": misaligned symbol " << sym
                   << " for relocation " << rel;
      check(val, lo, hi);
    };

    // Unlike other psABIs, the LoongArch ABI uses the same relocation
    // types to refer to GOT entries for thread-local symbols and regular
    // ones. Therefore, G may refer to a TLSGD or a regular GOT slot
    // depending on the symbol type.
    //
    // Note that as of August 2023, both GCC and Clang treat TLSLD relocs
    // as if they were TLSGD relocs for LoongArch, which is a clear bug.
    // We need to handle TLSLD relocs as synonyms for TLSGD relocs for the
    // sake of bug compatibility.
    auto get_got_idx = [&] {
      if (sym.has_tlsgd(ctx))
        return sym.get_tlsgd_idx(ctx);
      return sym.get_got_idx(ctx);
    };

    u64 S = sym.get_addr(ctx); // symval
    u64 A = rel.r_addend;
    u64 P = get_addr() + rel.r_offset; // get_addr(): output_section->shdr.sh_addr + offset; loc-P == ctx.buf(start address of OutputFile), i.e., loc is the absolute address of this reloc, P is the relative address of this reloc from the start of OutputFile.
    u64 G = get_got_idx() * sizeof(Word<E>);
    u64 GOT = ctx.got->shdr.sh_addr;

    switch (rel.r_type) {
    case R_LARCH_32:
      if constexpr (E::is_64)
        *(ul32 *)loc = S + A;
      else
        apply_dyn_absrel(ctx, sym, rel, loc, S, A, P, &dynrel);
      break;
    case R_LARCH_64:
      assert(E::is_64);
      apply_dyn_absrel(ctx, sym, rel, loc, S, A, P, &dynrel);
      break;
    case R_LARCH_B16:
      check_branch(S + A - P, -(1 << 17), 1 << 17);
      write_k16(loc, (S + A - P) >> 2);
      break;
    case R_LARCH_B21:
      check_branch(S + A - P, -(1 << 22), 1 << 22);
      write_d5k16(loc, (S + A - P) >> 2);
      break;
    case R_LARCH_B26: {
      i64 val = S + A - P;
      if (val < -(1 << 27) || (1 << 27) <= val)
        val = get_thunk_addr(i) + A - P;
      write_d10k16(loc, val >> 2);
      break;
    }
    case R_LARCH_ABS_LO12:
      write_k12(loc, S + A);
      break;
    case R_LARCH_ABS_HI20:
      write_j20(loc, (S + A) >> 12);
      break;
    case R_LARCH_ABS64_LO20:
      write_j20(loc, (S + A) >> 32);
      break;
    case R_LARCH_ABS64_HI12:
      write_k12(loc, (S + A) >> 52);
      break;
    case R_LARCH_PCALA_LO12:
      // It looks like R_LARCH_PCALA_LO12 is sometimes used for JIRL even
      // though the instruction takes a 16 bit immediate rather than 12 bits.
      // It is contrary to the psABI document, but GNU ld has special
      // code to handle it, so we accept it too.
      if ((*(ul32 *)loc & 0xfc00'0000) == 0x4c00'0000)
        write_k16(loc, sign_extend(S + A, 11) >> 2);
      else
        write_k12(loc, S + A); // put (S + A) into [11, 0]bits of addi.d, which is the part of si12 of 'addi.d rd, rj, si12'
      break;
    case R_LARCH_PCALA_HI20:
      write_j20(loc, hi20(S + A, P)); // put hi20(S + A, P) into [24, 5]bits of pcalau12i, which is the part of si20 of 'pcalau12i rd, si20'
      break;
    case R_LARCH_PCALA64_LO20:
      write_j20(loc, higher20(S + A, P));
      break;
    case R_LARCH_PCALA64_HI12:
      write_k12(loc, highest12(S + A, P));
      break;
    case R_LARCH_GOT_PC_LO12:
      write_k12(loc, GOT + G + A);
      break;
    case R_LARCH_GOT_PC_HI20:
      write_j20(loc, hi20(GOT + G + A, P));
      break;
    case R_LARCH_GOT64_PC_LO20:
      write_j20(loc, higher20(GOT + G + A, P));
      break;
    case R_LARCH_GOT64_PC_HI12:
      write_k12(loc, highest12(GOT + G + A, P));
      break;
    case R_LARCH_GOT_LO12:
      write_k12(loc, GOT + G + A);
      break;
    case R_LARCH_GOT_HI20:
      write_j20(loc, (GOT + G + A) >> 12);
      break;
    case R_LARCH_GOT64_LO20:
      write_j20(loc, (GOT + G + A) >> 32);
      break;
    case R_LARCH_GOT64_HI12:
      write_k12(loc, (GOT + G + A) >> 52);
      break;
    case R_LARCH_TLS_LE_LO12:
      write_k12(loc, S + A - ctx.tp_addr);
      break;
    case R_LARCH_TLS_LE_HI20:
      write_j20(loc, (S + A - ctx.tp_addr) >> 12);
      break;
    case R_LARCH_TLS_LE64_LO20:
      write_j20(loc, (S + A - ctx.tp_addr) >> 32);
      break;
    case R_LARCH_TLS_LE64_HI12:
      write_k12(loc, (S + A - ctx.tp_addr) >> 52);
      break;
    case R_LARCH_TLS_IE_PC_LO12:
      write_k12(loc, sym.get_gottp_addr(ctx) + A);
      break;
    case R_LARCH_TLS_IE_PC_HI20:
      write_j20(loc, hi20(sym.get_gottp_addr(ctx) + A, P));
      break;
    case R_LARCH_TLS_IE64_PC_LO20:
      write_j20(loc, higher20(sym.get_gottp_addr(ctx) + A, P));
      break;
    case R_LARCH_TLS_IE64_PC_HI12:
      write_k12(loc, highest12(sym.get_gottp_addr(ctx) + A, P));
      break;
    case R_LARCH_TLS_IE_LO12:
      write_k12(loc, sym.get_gottp_addr(ctx) + A);
      break;
    case R_LARCH_TLS_IE_HI20:
      write_j20(loc, (sym.get_gottp_addr(ctx) + A) >> 12);
      break;
    case R_LARCH_TLS_IE64_LO20:
      write_j20(loc, (sym.get_gottp_addr(ctx) + A) >> 32);
      break;
    case R_LARCH_TLS_IE64_HI12:
      write_k12(loc, (sym.get_gottp_addr(ctx) + A) >> 52);
      break;
    case R_LARCH_TLS_LD_PC_HI20:
    case R_LARCH_TLS_GD_PC_HI20:
      check(sym.get_tlsgd_addr(ctx) + A - P, -(1LL << 31), 1LL << 31);
      write_j20(loc, hi20(sym.get_tlsgd_addr(ctx) + A, P));
      break;
    case R_LARCH_TLS_LD_HI20:
    case R_LARCH_TLS_GD_HI20:
      write_j20(loc, (sym.get_tlsgd_addr(ctx) + A) >> 12);
      break;
    case R_LARCH_ADD6:
      *loc = (*loc & 0b1100'0000) | ((*loc + S + A) & 0b0011'1111);
      break;
    case R_LARCH_ADD8:
      *loc += S + A;
      break;
    case R_LARCH_ADD16:
      *(ul16 *)loc += S + A;
      break;
    case R_LARCH_ADD32:
      *(ul32 *)loc += S + A;
      break;
    case R_LARCH_ADD64:
      *(ul64 *)loc += S + A;
      break;
    case R_LARCH_SUB6:
      *loc = (*loc & 0b1100'0000) | ((*loc - S - A) & 0b0011'1111);
      break;
    case R_LARCH_SUB8:
      *loc -= S + A;
      break;
    case R_LARCH_SUB16:
      *(ul16 *)loc -= S + A;
      break;
    case R_LARCH_SUB32:
      *(ul32 *)loc -= S + A;
      break;
    case R_LARCH_SUB64:
      *(ul64 *)loc -= S + A;
      break;
    case R_LARCH_32_PCREL:
      *(ul32 *)loc = S + A - P;
      break;
    case R_LARCH_64_PCREL:
      *(ul64 *)loc = S + A - P;
      break;
    case R_LARCH_ADD_ULEB128:
      overwrite_uleb(loc, read_uleb(loc) + S + A);
      break;
    case R_LARCH_SUB_ULEB128:
      overwrite_uleb(loc, read_uleb(loc) - S - A);
      break;
    case R_LARCH_TLS_LE_HI20_R:
      write_j20(loc, (S + A + 0x800 - ctx.tp_addr) >> 12);
      break;
    case R_LARCH_TLS_LE_LO12_R:
      write_k12(loc, S + A - ctx.tp_addr);
      break;
    case R_LARCH_TLS_LE_ADD_R:
      break;
    case R_LARCH_PCREL20_S2: {
      const u32 pcaddi = 0x18000000;

      u32 pca = *(u32 *)loc;
      u32 rd = pca & 0x1f;
      u32 imm = (((S + A - P) >> 2) << 5) & 0x01ffffe0;
      pca = pcaddi | imm | rd;

      putl32(pca, loc);
      break;
    }
    default:
      unreachable();
    }
  }
}

template <>
void InputSection<E>::apply_reloc_nonalloc(Context<E> &ctx, u8 *base) {
  std::span<const ElfRel<E>> rels = get_rels(ctx);

  for (i64 i = 0; i < rels.size(); i++) {
    const ElfRel<E> &rel = rels[i];
    if (rel.r_type == R_NONE)
      continue;

    Symbol<E> &sym = *file.symbols[rel.r_sym];
    u8 *loc = base + rel.r_offset;

    if (!sym.file) {
      record_undef_error(ctx, rel);
      continue;
    }

    SectionFragment<E> *frag;
    i64 frag_addend;
    std::tie(frag, frag_addend) = get_fragment(ctx, rel);

    u64 S = frag ? frag->get_addr(ctx) : sym.get_addr(ctx);
    u64 A = frag ? frag_addend : (i64)rel.r_addend;

    switch (rel.r_type) {
    case R_LARCH_32:
      *(ul32 *)loc = S + A;
      break;
    case R_LARCH_64:
      if (std::optional<u64> val = get_tombstone(sym, frag))
        *(ul64 *)loc = *val;
      else
        *(ul64 *)loc = S + A;
      break;
    case R_LARCH_ADD6:
      *loc = (*loc & 0b1100'0000) | ((*loc + S + A) & 0b0011'1111);
      break;
    case R_LARCH_ADD8:
      *loc += S + A;
      break;
    case R_LARCH_ADD16:
      *(ul16 *)loc += S + A;
      break;
    case R_LARCH_ADD32:
      *(ul32 *)loc += S + A;
      break;
    case R_LARCH_ADD64:
      *(ul64 *)loc += S + A;
      break;
    case R_LARCH_SUB6:
      *loc = (*loc & 0b1100'0000) | ((*loc - S - A) & 0b0011'1111);
      break;
    case R_LARCH_SUB8:
      *loc -= S + A;
      break;
    case R_LARCH_SUB16:
      *(ul16 *)loc -= S + A;
      break;
    case R_LARCH_SUB32:
      *(ul32 *)loc -= S + A;
      break;
    case R_LARCH_SUB64:
      *(ul64 *)loc -= S + A;
      break;
    case R_LARCH_TLS_DTPREL32:
      if (std::optional<u64> val = get_tombstone(sym, frag))
        *(ul32 *)loc = *val;
      else
        *(ul32 *)loc = S + A - ctx.dtp_addr;
      break;
    case R_LARCH_TLS_DTPREL64:
      if (std::optional<u64> val = get_tombstone(sym, frag))
        *(ul64 *)loc = *val;
      else
        *(ul64 *)loc = S + A - ctx.dtp_addr;
      break;
    case R_LARCH_ADD_ULEB128:
      overwrite_uleb(loc, read_uleb(loc) + S + A);
      break;
    case R_LARCH_SUB_ULEB128:
      overwrite_uleb(loc, read_uleb(loc) - S - A);
      break;
    case R_LARCH_PCREL20_S2: {
      const u32 pcaddi = 0x18000000;

      u32 pca = *(u32 *)loc;
      u32 rd = pca & 0x1f;
      u32 imm = (((S + A - P) >> 2) << 5) & 0x01ffffe0;
      pca = pcaddi | imm |rd;

      putl32(pca, loc);
      break;
    }
    default:
      Fatal(ctx) << *this << ": invalid relocation for non-allocated sections: "
                 << rel;
      break;
    }
  }
}

template <>
void InputSection<E>::copy_contents_loongarch(Context<E> &ctx, u8 *buf) {
  // If a section is not relaxed, we can copy it as a one big chunk.
  if (extra.r_deltas.empty()) {
    copy_contents(ctx, buf);
    return;
  }

  // A relaxed section is copied piece-wise.
  std::span<const ElfRel<E>> rels = get_rels(ctx);
  i64 pos = 0;

  for (i64 i = 0; i < rels.size(); i++) {
    i64 delta = extra.r_deltas[i + 1] - extra.r_deltas[i]; // The number of bytes removed from current rel to next rel.
    if (delta == 0)
      continue;
    assert(delta > 0);

    const ElfRel<E> &r = rels[i];
    memcpy(buf, contents.data() + pos, r.r_offset - pos);
    buf += r.r_offset - pos;
    pos = r.r_offset + delta;
  }

  memcpy(buf, contents.data() + pos, contents.size() - pos);
}

template <>
void InputSection<E>::fix_roffset(Context<E> &ctx, u8 *buf) {
  if (extra.r_deltas.empty()) {
    return;
  }

  // Fix reloc's r_offset
  const ElfShdr<E> &shdr = file.elf_sections[relsec_idx];
  u8 *begin = file.mf->data + shdr.sh_offset;
  u8 *end = begin + shdr.sh_size;
  if (file.mf->data + file.mf->size < end)
    Fatal(ctx) << file << ": section header is out of range: " << shdr.sh_offset;

  u64 size = end - begin;
  u64 len = size / sizeof(ElfRel<E>);
  if (size % sizeof(ElfRel<E>))
    Fatal(ctx) << file << ": corrupted section";

  ElfRel<E> *rels = (ElfRel<E> *)begin;
  for (i64 i = 0; i < len; i++) {
    rels[i].r_offset -= extra.r_deltas[i];
  }
}

template <>
void InputSection<E>::scan_relocations(Context<E> &ctx) {
  assert(shdr().sh_flags & SHF_ALLOC);

  this->reldyn_offset = file.num_dynrel * sizeof(ElfRel<E>);
  std::span<const ElfRel<E>> rels = get_rels(ctx);

  // Scan relocations
  for (i64 i = 0; i < rels.size(); i++) {
    const ElfRel<E> &rel = rels[i];

    if (rel.r_type == R_NONE || rel.r_type == R_LARCH_RELAX ||
        rel.r_type == R_LARCH_MARK_LA || rel.r_type == R_LARCH_MARK_PCREL ||
        rel.r_type == R_LARCH_ALIGN)
      continue;

    if (record_undef_error(ctx, rel))
      continue;

    Symbol<E> &sym = *file.symbols[rel.r_sym];

    if (sym.is_ifunc())
      sym.flags |= NEEDS_GOT | NEEDS_PLT;

    switch (rel.r_type) {
    case R_LARCH_32:
      if constexpr (E::is_64)
        scan_absrel(ctx, sym, rel);
      else
        scan_dyn_absrel(ctx, sym, rel);
      break;
    case R_LARCH_64:
      assert(E::is_64);
      scan_dyn_absrel(ctx, sym, rel);
      break;
    case R_LARCH_B26:
    case R_LARCH_PCALA_HI20:
      if (sym.is_imported)
        sym.flags |= NEEDS_PLT;
      break;
    case R_LARCH_GOT_HI20:
    case R_LARCH_GOT_PC_HI20:
      sym.flags |= NEEDS_GOT;
      break;
    case R_LARCH_TLS_IE_HI20:
    case R_LARCH_TLS_IE_PC_HI20:
      sym.flags |= NEEDS_GOTTP;
      break;
    case R_LARCH_TLS_LD_PC_HI20:
    case R_LARCH_TLS_GD_PC_HI20:
    case R_LARCH_TLS_LD_HI20:
    case R_LARCH_TLS_GD_HI20:
      sym.flags |= NEEDS_TLSGD;
      break;
    case R_LARCH_32_PCREL:
    case R_LARCH_64_PCREL:
      scan_pcrel(ctx, sym, rel);
      break;
    case R_LARCH_TLS_LE_HI20:
    case R_LARCH_TLS_LE_LO12:
    case R_LARCH_TLS_LE64_LO20:
    case R_LARCH_TLS_LE64_HI12:
    case R_LARCH_TLS_LE_HI20_R:
    case R_LARCH_TLS_LE_LO12_R:
      check_tlsle(ctx, sym, rel);
      break;
    case R_LARCH_B16:
    case R_LARCH_B21:
    case R_LARCH_ABS_HI20:
    case R_LARCH_ABS_LO12:
    case R_LARCH_ABS64_LO20:
    case R_LARCH_ABS64_HI12:
    case R_LARCH_PCALA_LO12:
    case R_LARCH_PCALA64_LO20:
    case R_LARCH_PCALA64_HI12:
    case R_LARCH_GOT_PC_LO12:
    case R_LARCH_GOT64_PC_LO20:
    case R_LARCH_GOT64_PC_HI12:
    case R_LARCH_GOT_LO12:
    case R_LARCH_GOT64_LO20:
    case R_LARCH_GOT64_HI12:
    case R_LARCH_TLS_IE_PC_LO12:
    case R_LARCH_TLS_IE64_PC_LO20:
    case R_LARCH_TLS_IE64_PC_HI12:
    case R_LARCH_TLS_IE_LO12:
    case R_LARCH_TLS_IE64_LO20:
    case R_LARCH_TLS_IE64_HI12:
    case R_LARCH_ADD6:
    case R_LARCH_SUB6:
    case R_LARCH_ADD8:
    case R_LARCH_SUB8:
    case R_LARCH_ADD16:
    case R_LARCH_SUB16:
    case R_LARCH_ADD32:
    case R_LARCH_SUB32:
    case R_LARCH_ADD64:
    case R_LARCH_SUB64:
    case R_LARCH_ADD_ULEB128:
    case R_LARCH_SUB_ULEB128:
    case R_LARCH_TLS_LE_ADD_R:
      break;
    default:
      Error(ctx) << *this << ": unknown relocation: " << rel;
    }
  }
}

static bool is_resizable(InputSection<E> *isec) {
  return isec && isec->is_alive && (isec->shdr().sh_flags & SHF_ALLOC) &&
         (isec->shdr().sh_flags & SHF_EXECINSTR);
}

// Scan relocations to shrink sections. We do not delete bytes here, we just record how many bytes be deleted for each reloc, and adjust sec.sh_size. When we copy inputsection to outputsection, we only copy part of bytes, just like copy_contents_riscv. When relocate, we adjust the instruction bytes.
// OutputSections'shdrs have been set before.
static void shrink_section(Context<E> &ctx, InputSection<E> &isec) {
  // get the RelRels
  // NOTE: we need to change ElfRel itself so we can not use InputSection::get_rels(ctx);
  if (isec.relsec_idx == -1) // this section no need to relocate.
    return;

  ObjectFile<E> &file = isec.file;
  const ElfShdr<E> &shdr = file.elf_sections[isec.relsec_idx];
  u8 *begin = file.mf->data + shdr.sh_offset;
  u8 *end = begin + shdr.sh_size;
  if (file.mf->data + file.mf->size < end)
    Fatal(ctx) << file << ": section header is out of range: " << shdr.sh_offset;
  
  u64 size = end - begin;
  u64 len = size / sizeof(ElfRel<E>);
  if (size % sizeof(ElfRel<E>))
    Fatal(ctx) << file << ": corrupted section";
  ElfRel<E> *rels = (ElfRel<E> *)begin;

  isec.extra.r_deltas.resize(len + 1);

  i64 delta = 0; // The number of bytes deleted until current rel.

  for (i64 i = 0; i < len; i++) { // rels are ordered by r_offset
    ElfRel<E> &r = rels[i];
    Symbol<E> &sym = *isec.file.symbols[r.r_sym];
    u64 symval = sym.get_addr(ctx) + r.r_addend; // TODO(wx): if sym_sec == isec, we should consider the delta.
    u64 pc = isec.get_addr() + r.r_offset - delta;
    InputSection<E> *sym_sec = sym.get_input_section();
    isec.extra.r_deltas[i] = delta;

    // Handling R_LARCH_ALIGN is mandatory.
    //
    // R_LARCH_ALIGN refers to NOP instructions. We need to eliminate some
    // or all of the instructions so that the instruction that immediately
    // follows the NOPs is aligned to a specified alignment boundary.

    /* if (r.r_type == R_LARCH_ALIGN) { // TODO(wx): we need to make sure that no other relaxation applied after align relaxation */
    /*   // The total bytes of NOPs is stored to r_addend, so the next */
    /*   // instruction is r_addend away. */
    /*   u64 loc = isec.get_addr() + r.r_offset - delta; // NOTE: we can not adjust r_offset, as relocate and copy_contents_loongarch both use old r_offset. */
    /*   u64 addend, alignment, max = 0; */
    /*   /1* For R_LARCH_ALIGN, symval is sec_addr (sec) + rel->r_offset */
	 /* + (alingmeng - 4). */
	 /* If r_symndx is 0, alignmeng-4 is r_addend. */
	 /* If r_symndx > 0, alignment-4 is 2^(r_addend & 0xff)-4.  *1/ */
    /*   if (r.r_sym > 0) { */
    /*       alignment = 1 << (r.r_addend & 0xff); */
    /*       max = r.r_addend >> 8; */
    /*   } */
    /*   else */
    /*       alignment = r.r_addend + 4; */
    /*   addend = alignment - 4; /1* The bytes of NOPs added by R_LARCH_ALIGN.  *1/ */
    /*   u64 next_loc = loc + addend; */
    /*   assert(alignment <= (1 << isec.p2align)); */
    /*   u64 aligned_addr = ((loc - 1) & ~(alignment - 1)) + alignment; */
    /*   u64 need_nop_bytes = aligned_addr - loc; /1* *1/ */

    /*   if (addend < need_nop_bytes) { */
    /*     Error(ctx) << file << ": align relax, " << need_nop_bytes */
    /*                << " bytes required for alignment to " << alignment */
    /*                << "-byte boundary, but only " << addend << " present"; */
    /*   } */

    /*   r.r_type = R_LARCH_NONE; */

    /*   /1* If skipping more bytes than the specified maximum, */
    /*      then the alignment is not done at all and delete all NOPs.  *1/ */
    /*   if (max > 0 && need_nop_bytes > max) */
    /*       delta += addend; */
    /*   else */
    /*       delta += addend - need_nop_bytes; */
    /*   continue; */
    /* } */

    // Handling other relocations is optional.
    if (!ctx.arg.relax || i == len - 1 ||
        rels[i + 1].r_type != R_LARCH_RELAX)
      continue;

    // Linker-synthesized symbols haven't been assigned their final
    // values when we are shrinking sections because actual values can
    // be computed only after we fix the file layout. Therefore, we
    // assume that relocations against such symbols are always
    // non-relaxable.
    if (sym.file == ctx.internal_obj)
      continue;


    switch (r.r_type) {
    case R_LARCH_PCALA_HI20: {
      if ((i + 4) > len)
        continue;

      u32 pca = *(u32 *)(isec.contents.data() + r.r_offset); // do not sub delta, as the contents has not been moved yet.
      u32 add = *(u32 *)(isec.contents.data() + rels[i+2].r_offset);
      u32 rd = pca & 0x1f;
      u64 max_alignment = 0;
      for(int i = 0; i < ctx.chunks.size(); i++) {
        OutputSection<E> *osec = ctx.chunks[i]->to_osec();
        if (osec)
            max_alignment = (u64)(osec->shdr.sh_addralign) > max_alignment ?
                (u64)(osec->shdr.sh_addralign) : max_alignment;
      }
      if (sym_sec->shdr().sh_flags & SHF_WRITE) {
        max_alignment = LOONGARCH_MAX_PAGESIZE > max_alignment ? LOONGARCH_MAX_PAGESIZE : max_alignment;
        if (symval > pc)
          pc -= max_alignment;
        else if (symval < pc)
          pc += max_alignment;
      } else {
        if (symval > pc)
          pc -= max_alignment;
        else if (symval < pc)
          pc += max_alignment;
      }

      const u32 addi_d = 0x02c00000;

      /* Is pcalau12i + addi.d insns?  */
      if (rels[i+2].r_type != R_LARCH_PCALA_LO12
          || rels[i+1].r_type != R_LARCH_RELAX
          || rels[i+3].r_type != R_LARCH_RELAX
          || r.r_offset + 4 != rels[i+2].r_offset
          || (add & addi_d) != addi_d
          /* Is pcalau12i $rd + addi.d $rd,$rd?  */
          || (add & 0x1f) != rd
          || ((add >> 5) & 0x1f) != rd
          /* Can be relaxed to pcaddi?  */
          || symval & 0x3 /* 4 bytes align.  */
          || (long)(symval - pc) < (long)(int32_t)0xffe00000
          || (long)(symval - pc) > (long)(int32_t)0x1ffffc)
        continue;

      r.r_type = R_LARCH_PCREL20_S2;

      rels[i+2].r_sym = 0;
      rels[i+2].r_type = R_LARCH_NONE;

      // NOTE: we should set delta of all rels of this symbol, and the index
      isec.extra.r_deltas[i+1] = isec.extra.r_deltas[i+2] = isec.extra.r_deltas[i+3] = delta;
      delta += 4;
      i += 3;
      break;
    }
    }
  }

  isec.extra.r_deltas[len] = delta;
  isec.sh_size -= delta;
}

template <>
i64 loongarch_resize_sections<E>(Context<E> &ctx) {
  Timer t(ctx, "loongarch_resize_sections");

  // Find all the relocations that can be relaxed.
  // This step should only shrink sections.
  tbb::parallel_for_each(ctx.objs, [&](ObjectFile<E> *file) {
    for (std::unique_ptr<InputSection<E>> &isec : file->sections)
      if (is_resizable(isec.get()))
        shrink_section(ctx, *isec);
  });

  // Fix symbol values.
  tbb::parallel_for_each(ctx.objs, [&](ObjectFile<E> *file) {
    for (Symbol<E> *sym : file->symbols) {
      if (sym->file != file)
        continue;

      InputSection<E> *isec = sym->get_input_section();
      if (!isec || isec->extra.r_deltas.empty())
        continue;

      std::span<const ElfRel<E>> rels = isec->get_rels(ctx);
      auto it = std::lower_bound(rels.begin(), rels.end(), sym->value,
                                 [&](const ElfRel<E> &r, u64 val) { // rels are ordered by r_offset
        return r.r_offset < val;
      });

      sym->value -= isec->extra.r_deltas[it - rels.begin()];
    }
  });

  // Re-compute section offset again to finalize them.
  compute_section_sizes(ctx);
  return set_osec_offsets(ctx);
}

template <>
void Thunk<E>::copy_buf(Context<E> &ctx) {
  static const ul32 insn[] = {
    0x1e00'000c, // pcaddu18i $t0, 0
    0x4c00'0180, // jirl      $zero, $t0, 0
  };

  static_assert(E::thunk_size == sizeof(insn));

  u8 *buf = ctx.buf + output_section.shdr.sh_offset + offset;
  u64 P = output_section.shdr.sh_addr + offset;

  for (Symbol<E> *sym : symbols) {
    u64 S = sym->get_addr(ctx);

    memcpy(buf, insn, sizeof(insn));
    write_j20(buf, (S - P + 0x20000) >> 18);
    write_k16(buf + 4, (S - P) >> 2);

    buf += sizeof(insn);
    P += sizeof(insn);
  }
}

} // namespace mold::elf

#endif
