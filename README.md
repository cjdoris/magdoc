# magdoc

This is an experimental documentation generator for the [MAGMA computer algebra system](http://magma.maths.usyd.edu.au/magma).

## Usage

Install the Tatsu package (e.g. `pip install tatsu`) used for parsing.

Run the Python 3 script `magdoc.py` with the source files as arguments. It will output some files in markdown format containing documentation. Example:

```
$ python3 magdoc.py -o output_dir source.mag another_source.mag
```

## Inputs

The source files are either:
- MAGMA source files (extension `.magma`, `.mag` or `.m`)
- Markdown files (extension `.markdown` or `.md`)

For MAGMA files, line comments beginning `///` (but not `////`) are special *doc comments* with meanings described below. Markdown files are interpreted as if each line is a doc comment (without the leading `///`).

If a filename starts with `///`, it is interpreted as a single doc comment; this is useful to put files into sections without editing the source.

- `/// TEXT` is markdown formatted documentation (note the space before `TEXT`). If they appear immediately after a sectioning command (with no intervening space) then they document the section; otherwise they document the next definition (usually an intrinsic).
- `///# NAME` define sections. Additional `#`s give deeper sections.
- `///hide` hides the thing to which it is attached.
- `///hide-all` hides everything until `///hide-none` is encountered.
- `///ditto` groups this intrinsic with the previous one; typically they have similar inputs and the same return types; when the return types are the same, they are merged in the documentation.
- `///priority N` sets the priority of the thing to which it is attached to the number `N`. Things with higher priority appear first, and the default priority is 0.
- `///toc` inserts a simple table of contents of the current section.
- `///param NAME:=DEFAULT TEXT` documents the parameter NAME. `:=DEFAULT` and `TEXT` are both optional. `DEFAULT` must not contain any spaces.
- `///label NAME` after a section command associates a label to the section.
- `///sec NAME` changes the section to the one with the label `NAME`.
- `///~` is ignored by the parser, so the rest of the line is treated as MAGMA code. This allows us to document intrinsics defined only in comments.

If an intrinsic does not have any documentation attached to it via doc comments, then we consider the `{}`-delimited docstring instead. If it is `{"}` then the intrinsic is grouped with the previous (the same as for `///ditto`), otherwise the contents become documentation for the intrinsic.

Example magma file:
```
///# My module
///This documents the first section.
///
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

/// This documents both Cat and Dog together.
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

///~intrinsic Fish() -> []
///~  {This documents Fish, even though it is in a comment.}
///~end intrinsic

//////////////////////////////////////
// four or more slashes are ignored //
//////////////////////////////////////
```

## Development

The main parser is in `magmaparser.py` which is generated using the Tatsu python package from `magma.ebnf` with the command `python3 -m tatsu -o magmaparser.py magma.ebnf`.

## License

Copyright (C) 2018 Christopher Doris

This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this.  If not, see http://www.gnu.org/licenses/.
