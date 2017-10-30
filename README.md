# magdoc

This is an experimental documentation generator for the [MAGMA computer algebra system](http://magma.maths.usyd.edu.au/magma).

## Usage

Run the Python 3 script `magdoc.py` with the source files as arguments. It will output some files in markdown format containing documentation. Example:

```
$ python3 magdoc.py source.mag another_source.mag
```

## Inputs

The source files are either:
- MAGMA source files (extension `.magma`, `.mag` or `.m`)
- Markdown files (extension `.markdown` or `.md`)

For MAGMA files, line comments beginning `///` (but not `////`) are special *doc comments* with meanings described below. Markdown files are interpreted as if they had `///` prepended to each line and then interpreted as a MAGMA file.

- Comments starting `/// ` (i.e. a space after the three slashes) are markdown formatted documentation. If they appear immediately after a sectioning command (with no intervening space) then they document the section; otherwise they document the next definition (usually an intrinsic).
- Comments starting `///#` are sectioning commands. Additional `#`s give deeper sections, and the rest is the section name.
- `///hide` hides the thing to which it is attached.
- `///ditto` groups this intrinsic with the previous one; typically they have similar inputs and the same return types; when the return types are the same, they are merged in the documentation.
- `///toc` inserts a simple table of contents of the current section.

If an intrinsic does not have any documentation attached to it via doc comments, then we consider the `{}`-delimited docstring instead. If it is `{"}` then the intrinsic is grouped with the previous (the same as for `///ditto`), otherwise the contents become documentation for the intrinsic.

Example magma file:
```
///# My module
///This documents the first section.

///toc

///## Introduction
/// This documents the introduction.
///
/// More information about the introduction

///## Main intrinsics

/// This documents Aardvark.
intrinsic Aardvark() -> []
  {This is ignored.}
end intrinsic;

intrinsic Bear() -> []
  {This documents Bear.}
end intrinsic;

/// This documents Cat and Doc.
intrinsic Cat() -> []
  {Ignored.}
end intrindic;

intrinsic Dog() -> []
  {"}
end intrinsic;

///hide
intrinsic Elephant() -> []
  {Elephant is undocumented.}
end intrinsic;
```
