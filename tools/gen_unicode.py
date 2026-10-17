import sys
import unicodedata

MAXCP = 0x110000


def ranges(pred):
    out, start = [], None
    for cp in range(MAXCP):
        hit = pred(cp)
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAXCP - 1))
    return out


def cat(cp):
    return unicodedata.category(chr(cp))


def is_letter(cp):
    return cat(cp).startswith("L")


def is_num(cp):
    return cat(cp).startswith("N")


def is_ws(cp):
    return cp in (9, 10, 11, 12, 13, 0x85) or cat(cp) in ("Zs", "Zl", "Zp")


def is_digit(cp):
    return cat(cp) == "Nd"


def ccc():
    return [(cp, unicodedata.combining(chr(cp))) for cp in range(MAXCP) if unicodedata.combining(chr(cp))]


def decomp():
    out = []
    for cp in range(MAXCP):
        if 0xAC00 <= cp <= 0xD7A3:
            continue
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue
        cps = [int(x, 16) for x in d.split()]
        out.append((cp, cps))
    return out


def comps(dec):
    out = []
    for cp, cps in dec:
        if len(cps) != 2:
            continue
        if unicodedata.normalize("NFC", chr(cp)) != chr(cp):
            continue
        if unicodedata.combining(chr(cp)) or unicodedata.combining(chr(cps[0])):
            continue
        out.append((cps[0], cps[1], cp))
    return sorted(out)


def emit(fh, name, rng):
    fh.write(f"constexpr Range k{name}[] = {{\n")
    for a, b in rng:
        fh.write(f"    {{0x{a:X}, 0x{b:X}}},\n")
    fh.write("};\n\n")


def main(path):
    dec = decomp()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f'constexpr const char* kUnicodeVersion = "{unicodedata.unidata_version}";\n\n')
        emit(fh, "Letter", ranges(is_letter))
        emit(fh, "Number", ranges(is_num))
        emit(fh, "Space", ranges(is_ws))
        emit(fh, "Digit", ranges(is_digit))
        fh.write("constexpr Ccc kCcc[] = {\n")
        for cp, c in ccc():
            fh.write(f"    {{0x{cp:X}, {c}}},\n")
        fh.write("};\n\nconstexpr Dec kDec[] = {\n")
        for cp, cps in dec:
            fh.write(f"    {{0x{cp:X}, 0x{cps[0]:X}, 0x{cps[1] if len(cps) > 1 else 0:X}}},\n")
        fh.write("};\n\nconstexpr Comp kComp[] = {\n")
        for a, b, c in comps(dec):
            fh.write(f"    {{0x{a:X}, 0x{b:X}, 0x{c:X}}},\n")
        fh.write("};\n")


if __name__ == "__main__":
    main(sys.argv[1])
