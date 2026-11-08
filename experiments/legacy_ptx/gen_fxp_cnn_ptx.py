#!/usr/bin/env python3
"""Bespoke PTX producer for one fixed CNN. Not the 2020 LLVM/SPIR-V compiler.

Network: conv 3x3x4 valid, integer ReLU, 2x2 max-pool, FC 676x10.
Type fxp_32_12_3 is width 32 and exponent -12. Mul does not return the sum
exponent -24. The shift below is the half-away-from-zero requantize to -12.
"""
import os
import sys

MUL = "_Z7fxp_mul12fxp_32_12_3_12fxp_32_12_3_"
ADD = "_Z7fxp_add12fxp_32_12_3_12fxp_32_12_3_"


class Ptx(object):
    def __init__(self):
        self.lines = []

    def add(self, text):
        self.lines.append(text)

    def call_bin(self, op, dst, a, b):
        self.add("{")
        self.add(".param .b32 p0;")
        self.add(".param .b32 p1;")
        self.add(".param .b32 pret;")
        self.add("st.param.b32 [p0+0], " + a + ";")
        self.add("st.param.b32 [p1+0], " + b + ";")
        self.add("call.uni (pret), " + op + ", (p0, p1);")
        self.add("ld.param.b32 " + dst + ", [pret+0];")
        self.add("}")

    def requant(self, tag):
        # %rprod -> %ralign. Clobbers %r0-%r3 and %p0-%p2.
        self.add("mov.b32 %r0, 0x80000000;")
        self.add("setp.eq.s32 %p0, %rprod, %r0;")
        self.add("@%p0 bra rq_min_" + tag + ";")
        self.add("setp.lt.s32 %p1, %rprod, 0;")
        self.add("mov.s32 %r1, %rprod;")
        self.add("@%p1 sub.s32 %r1, 0, %rprod;")
        self.add("and.b32 %r2, %r1, 4095;")
        self.add("shr.u32 %r3, %r1, 12;")
        self.add("setp.ge.u32 %p2, %r2, 2048;")
        self.add("@%p2 add.s32 %r3, %r3, 1;")
        self.add("mov.s32 %ralign, %r3;")
        self.add("@%p1 sub.s32 %ralign, 0, %r3;")
        self.add("bra rq_done_" + tag + ";")
        self.add("rq_min_" + tag + ":")
        self.add("sub.s32 %ralign, 0, 524288;")
        self.add("rq_done_" + tag + ":")

    def load32(self, dst, base, index):
        self.add("mul.lo.u32 %r0, " + index + ", 4;")
        self.add("cvt.u64.u32 %rd4, %r0;")
        self.add("add.u64 %rd4, " + base + ", %rd4;")
        self.add("ld.global.s32 " + dst + ", [%rd4];")

    def store32(self, base, index, src):
        self.add("mul.lo.u32 %r0, " + index + ", 4;")
        self.add("cvt.u64.u32 %rd4, %r0;")
        self.add("add.u64 %rd4, " + base + ", %rd4;")
        self.add("st.global.s32 [%rd4], " + src + ";")


def stub(name):
    return (
        ".func (.param .b32 func_retval0) %s(.param .b32 lhs, .param .b32 rhs)\n"
        "{ .reg .b32 %%z; mov.u32 %%z, 0; st.param.b32 [func_retval0+0], %%z; ret; }\n"
        % name
    )


def emit_pairs(p, name, do_requant):
    p.add(".visible .entry %s(.param .u64 a, .param .u64 b, .param .u64 out, .param .u32 n)" % name)
    p.add("{")
    p.add(".reg .pred %p0, %p1, %p2, %p3;")
    p.add(".reg .b32 %r0, %r1, %r2, %r3, %ri, %rn, %ra, %rb, %rprod, %ralign;")
    p.add(".reg .b64 %rd0, %rd1, %rd2, %rd4, %rdt;")
    p.add("ld.param.u64 %rdt, [a];")
    p.add("cvta.to.global.u64 %rd0, %rdt;")
    p.add("ld.param.u64 %rdt, [b];")
    p.add("cvta.to.global.u64 %rd1, %rdt;")
    p.add("ld.param.u64 %rdt, [out];")
    p.add("cvta.to.global.u64 %rd2, %rdt;")
    p.add("ld.param.u32 %rn, [n];")
    p.add("mov.u32 %ri, 0;")
    p.add("%s_loop:" % name)
    p.add("setp.ge.u32 %p3, %ri, %rn;")
    p.add("@%p3 bra " + name + "_done;")
    p.load32("%ra", "%rd0", "%ri")
    p.load32("%rb", "%rd1", "%ri")
    p.call_bin(MUL, "%rprod", "%ra", "%rb")
    stored = "%rprod"
    if do_requant:
        p.requant(name)
        stored = "%ralign"
    p.store32("%rd2", "%ri", stored)
    p.add("add.u32 %ri, %ri, 1;")
    p.add("bra %s_loop;" % name)
    p.add("%s_done:" % name)
    p.add("ret;")
    p.add("}")
    p.add("")


def emit_conv(p, name, full):
    y_lim = "26" if full else "1"
    x_lim = "26" if full else "1"
    oc_lim = "4" if full else "1"
    p.add(".visible .entry %s(.param .u64 img, .param .u64 w, .param .u64 bias, .param .u64 out)" % name)
    p.add("{")
    p.add(".reg .pred %p0, %p1, %p2, %py, %px, %poc, %pky, %pkx;")
    p.add(".reg .b32 %r0, %r1, %r2, %r3, %ry, %rx, %roc, %rky, %rkx;")
    p.add(".reg .b32 %racc, %rpix, %rw, %rprod, %ralign, %ridx;")
    p.add(".reg .b64 %rd0, %rd1, %rd2, %rd3, %rd4, %rdt;")
    p.add("ld.param.u64 %rdt, [img];")
    p.add("cvta.to.global.u64 %rd0, %rdt;")
    p.add("ld.param.u64 %rdt, [w];")
    p.add("cvta.to.global.u64 %rd1, %rdt;")
    p.add("ld.param.u64 %rdt, [bias];")
    p.add("cvta.to.global.u64 %rd2, %rdt;")
    p.add("ld.param.u64 %rdt, [out];")
    p.add("cvta.to.global.u64 %rd3, %rdt;")
    p.add("mov.u32 %ry, 0;")
    p.add("%s_y:" % name)
    p.add("setp.ge.u32 %py, %ry, " + y_lim + ";")
    p.add("@%py bra " + name + "_ydone;")
    p.add("mov.u32 %rx, 0;")
    p.add("%s_x:" % name)
    p.add("setp.ge.u32 %px, %rx, " + x_lim + ";")
    p.add("@%px bra " + name + "_xdone;")
    p.add("mov.u32 %roc, 0;")
    p.add("%s_oc:" % name)
    p.add("setp.ge.u32 %poc, %roc, " + oc_lim + ";")
    p.add("@%poc bra " + name + "_ocdone;")
    p.load32("%r1", "%rd2", "%roc")
    p.add("mov.s32 %racc, %r1;")
    p.add("mov.u32 %rky, 0;")
    p.add("%s_ky:" % name)
    p.add("setp.ge.u32 %pky, %rky, 3;")
    p.add("@%pky bra " + name + "_kydone;")
    p.add("mov.u32 %rkx, 0;")
    p.add("%s_kx:" % name)
    p.add("setp.ge.u32 %pkx, %rkx, 3;")
    p.add("@%pkx bra " + name + "_kxdone;")
    # pixel (y + ky) * 28 + (x + kx)
    p.add("add.u32 %ridx, %ry, %rky;")
    p.add("mul.lo.u32 %ridx, %ridx, 28;")
    p.add("add.u32 %ridx, %ridx, %rx;")
    p.add("add.u32 %ridx, %ridx, %rkx;")
    p.load32("%rpix", "%rd0", "%ridx")
    # weight oc * 9 + ky * 3 + kx
    p.add("mul.lo.u32 %ridx, %roc, 9;")
    p.add("mul.lo.u32 %r2, %rky, 3;")
    p.add("add.u32 %ridx, %ridx, %r2;")
    p.add("add.u32 %ridx, %ridx, %rkx;")
    p.load32("%rw", "%rd1", "%ridx")
    p.call_bin(MUL, "%rprod", "%rpix", "%rw")
    p.requant(name)
    p.call_bin(ADD, "%racc", "%racc", "%ralign")
    p.add("add.u32 %rkx, %rkx, 1;")
    p.add("bra %s_kx;" % name)
    p.add("%s_kxdone:" % name)
    p.add("add.u32 %rky, %rky, 1;")
    p.add("bra %s_ky;" % name)
    p.add("%s_kydone:" % name)
    p.add("setp.lt.s32 %p0, %racc, 0;")
    p.add("@%p0 mov.s32 %racc, 0;")
    p.add("mul.lo.u32 %ridx, %ry, 26;")
    p.add("add.u32 %ridx, %ridx, %rx;")
    p.add("mul.lo.u32 %ridx, %ridx, 4;")
    p.add("add.u32 %ridx, %ridx, %roc;")
    p.store32("%rd3", "%ridx", "%racc")
    p.add("add.u32 %roc, %roc, 1;")
    p.add("bra %s_oc;" % name)
    p.add("%s_ocdone:" % name)
    p.add("add.u32 %rx, %rx, 1;")
    p.add("bra %s_x;" % name)
    p.add("%s_xdone:" % name)
    p.add("add.u32 %ry, %ry, 1;")
    p.add("bra %s_y;" % name)
    p.add("%s_ydone:" % name)
    p.add("ret;")
    p.add("}")
    p.add("")


def emit_pool(p):
    p.add(".visible .entry kpool(.param .u64 src, .param .u64 dst)")
    p.add("{")
    p.add(".reg .pred %py, %px, %poc, %p0;")
    p.add(".reg .b32 %r0, %ry, %rx, %roc, %ridx, %ra, %rb, %rc, %rv, %rm;")
    p.add(".reg .b64 %rd0, %rd1, %rd4, %rdt;")
    p.add("ld.param.u64 %rdt, [src];")
    p.add("cvta.to.global.u64 %rd0, %rdt;")
    p.add("ld.param.u64 %rdt, [dst];")
    p.add("cvta.to.global.u64 %rd1, %rdt;")
    p.add("mov.u32 %ry, 0;")
    p.add("pool_y:")
    p.add("setp.ge.u32 %py, %ry, 13;")
    p.add("@%py bra pool_ydone;")
    p.add("mov.u32 %rx, 0;")
    p.add("pool_x:")
    p.add("setp.ge.u32 %px, %rx, 13;")
    p.add("@%px bra pool_xdone;")
    p.add("mov.u32 %roc, 0;")
    p.add("pool_oc:")
    p.add("setp.ge.u32 %poc, %roc, 4;")
    p.add("@%poc bra pool_ocdone;")
    # base = ((y*2)*26 + (x*2))*4 + oc
    p.add("mul.lo.u32 %ridx, %ry, 208;")
    p.add("mul.lo.u32 %r0, %rx, 8;")
    p.add("add.u32 %ridx, %ridx, %r0;")
    p.add("add.u32 %ridx, %ridx, %roc;")
    p.load32("%ra", "%rd0", "%ridx")
    p.add("add.u32 %ridx, %ridx, 4;")
    p.load32("%rb", "%rd0", "%ridx")
    p.add("add.u32 %ridx, %ridx, 100;")
    p.load32("%rc", "%rd0", "%ridx")
    p.add("add.u32 %ridx, %ridx, 4;")
    p.load32("%rv", "%rd0", "%ridx")
    p.add("mov.s32 %rm, %ra;")
    p.add("setp.gt.s32 %p0, %rb, %rm;")
    p.add("@%p0 mov.s32 %rm, %rb;")
    p.add("setp.gt.s32 %p0, %rc, %rm;")
    p.add("@%p0 mov.s32 %rm, %rc;")
    p.add("setp.gt.s32 %p0, %rv, %rm;")
    p.add("@%p0 mov.s32 %rm, %rv;")
    p.add("mul.lo.u32 %ridx, %ry, 13;")
    p.add("add.u32 %ridx, %ridx, %rx;")
    p.add("mul.lo.u32 %ridx, %ridx, 4;")
    p.add("add.u32 %ridx, %ridx, %roc;")
    p.store32("%rd1", "%ridx", "%rm")
    p.add("add.u32 %roc, %roc, 1;")
    p.add("bra pool_oc;")
    p.add("pool_ocdone:")
    p.add("add.u32 %rx, %rx, 1;")
    p.add("bra pool_x;")
    p.add("pool_xdone:")
    p.add("add.u32 %ry, %ry, 1;")
    p.add("bra pool_y;")
    p.add("pool_ydone:")
    p.add("ret;")
    p.add("}")
    p.add("")


def emit_fc(p):
    p.add(".visible .entry kfc(.param .u64 act, .param .u64 w, .param .u64 bias, .param .u64 out)")
    p.add("{")
    p.add(".reg .pred %p0, %p1, %p2, %pj, %pi;")
    p.add(".reg .b32 %r0, %r1, %r2, %r3, %rj, %ri, %ridx;")
    p.add(".reg .b32 %racc, %ract, %rw, %rprod, %ralign;")
    p.add(".reg .b64 %rd0, %rd1, %rd2, %rd3, %rd4, %rdt;")
    p.add("ld.param.u64 %rdt, [act];")
    p.add("cvta.to.global.u64 %rd0, %rdt;")
    p.add("ld.param.u64 %rdt, [w];")
    p.add("cvta.to.global.u64 %rd1, %rdt;")
    p.add("ld.param.u64 %rdt, [bias];")
    p.add("cvta.to.global.u64 %rd2, %rdt;")
    p.add("ld.param.u64 %rdt, [out];")
    p.add("cvta.to.global.u64 %rd3, %rdt;")
    p.add("mov.u32 %rj, 0;")
    p.add("fc_j:")
    p.add("setp.ge.u32 %pj, %rj, 10;")
    p.add("@%pj bra fc_jdone;")
    p.load32("%r1", "%rd2", "%rj")
    p.add("mov.s32 %racc, %r1;")
    p.add("mov.u32 %ri, 0;")
    p.add("fc_i:")
    p.add("setp.ge.u32 %pi, %ri, 676;")
    p.add("@%pi bra fc_idone;")
    p.load32("%ract", "%rd0", "%ri")
    p.add("mul.lo.u32 %ridx, %ri, 10;")
    p.add("add.u32 %ridx, %ridx, %rj;")
    p.load32("%rw", "%rd1", "%ridx")
    p.call_bin(MUL, "%rprod", "%ract", "%rw")
    p.requant("fc")
    p.call_bin(ADD, "%racc", "%racc", "%ralign")
    p.add("add.u32 %ri, %ri, 1;")
    p.add("bra fc_i;")
    p.add("fc_idone:")
    p.store32("%rd3", "%rj", "%racc")
    p.add("add.u32 %rj, %rj, 1;")
    p.add("bra fc_j;")
    p.add("fc_jdone:")
    p.add("ret;")
    p.add("}")
    p.add("")


def main():
    out = sys.argv[1] if len(sys.argv) == 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fxp_cnn_image.ptx")
    p = Ptx()
    p.add("// BESPOKE PRODUCER. Not the 2020 LLVM/SPIR-V compiler.")
    p.add("// Fixed conv 3x3x4 / integer ReLU / max-pool 2x2 / FC 676x10.")
    p.add("// fxp_32_12_3: width 32, exponent -12. Product exponent -24 is not returned.")
    p.add("// shr 12, half away from zero, then fxp_add at exponent -12.")
    p.add(".version 4.3")
    p.add(".target sm_20")
    p.add(".address_size 64")
    p.add("")
    p.add(stub(MUL))
    p.add(stub(ADD))
    emit_pairs(p, "kprobe", False)
    emit_pairs(p, "kreq", True)
    emit_conv(p, "ktile", False)
    emit_conv(p, "kconv", True)
    emit_pool(p)
    emit_fc(p)
    text = "\n".join(p.lines) + "\n"
    with open(out, "w") as f:
        f.write(text)
    print(out, "lines", text.count("\n"))


if __name__ == "__main__":
    main()
