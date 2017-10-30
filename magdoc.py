from magmaparser import MagmaParser
import re
from itertools import groupby
import string

re_ident = re.compile('^(?!(_|adj|and|assert|assert2|assert3|assigned|break|by|case|cat|catch|clear|cmpeq|cmpne|continue|declare|default|delete|diff|div|do|elif|else|end|eq|error|eval|exists|exit|false|for|forall|forward|fprintf|freeze|function|ge|gt|if|iload|import|in|intrinsic|is|join|le|load|local|lt|meet|mod|ne|not|notadj|notin|notsubset|or|print|printf|procedure|quit|random|read|readi|rep|repeat|require|requirege|requirerange|restore|return|save|sdiff|select|subset|then|time|to|true|try|until|vprint|vprintf|vtime|when|where|while|xor)$)[A-Za-z_][A-Za-z0-9_]*$')

DOC_CHAR = '/'
NOT_DOC_CHAR = '/'
SEC_CHAR = '#'
TEXT_CHAR = ' '

MAGMA_EXTNS = {'.magma','.mag','.m'}
MARKDOWN_EXTNS = {'.markdown','.md'}

class Formatter(string.Formatter):
  def format_field(self, x, spec):
    if isinstance(x, list) and spec.startswith('~'):
      return spec[1:].join(x)
    elif isinstance(x, int) and spec.startswith('*'):
      return x * spec[1:]
    elif spec.startswith('?'):
      return spec[1:] if x else ''
    elif spec == '':
      assert isinstance(x, str)
      return x
    else:
      raise ValueError('bad template')

FORMATTER = Formatter()
def template_subs(tmp, *args, **kwargs):
  return FORMATTER.format(tmp, *args, **kwargs)

class Node:
  def __init__(self, ast):
    self.ast = ast
    self.src_order = None
    self.src_filename = None
    self.docs = []
    self.section = None
  def __repr__(self):
    return self.__class__.__name__

class Statement(Node):
  pass

class IntrinsicStatement(Statement):
  def __init__(self, ast):
    super().__init__(ast)
    self.name = tr_identifier(ast.name)
    self.args = [tr_intrinsic_argument(x) for x in ast.args] if ast.args else []
    self.params = [tr_parameter(x) for x in ast.params] if ast.params else []
    self.returns = [tr_scategory(x) for x in ast.returns] if ast.returns else []
    self.doc = tr_intrinsic_doc(ast.doc)
  def decl_code(self, returns=True):
    ret = 'intrinsic '
    ret += self.name.code()
    ret += ' ('
    ret += ', '.join(arg.code() for arg in self.args)
    if self.params:
      if self.args: ret += ' '
      ret += ': '
      ret += ', '.join(param.code() for param in self.params)
    ret += ')'
    if returns and self.returns:
      ret += ' -> '
      ret += ', '.join(r.code() for r in self.returns)
    return ret

class Identifier(Node):
  def __init__(self, ast):
    super().__init__(ast)
    self.value = ast.value
  def code(self):
    if re_ident.match(self.value):
      return self.value
    else:
      return '\'' + self.value + '\''

class IntrinsicArgument(Node):
  def __init__(self, ast):
    super().__init__(ast)
    self.is_ref = True if ast.ref else False
    self.name = tr_identifier(ast.name)
    self.cat = tr_category(ast.cat)
  def code(self):
    return '{ref}{name}{cat}'.format(
      ref = '~' if self.is_ref else '',
      name = self.name.code(),
      cat = (' :: ' + self.cat.code()) if (self.cat and not isinstance(self.cat, AnyCategory)) else ''
    )

class _Char(Node):
  def __init__(self, ast):
    super().__init__(ast)
    self.value = ast.value

class EscapedChar(_Char):
  def code(self):
    return '\\' + self.value

class Char(_Char):
  def code(self):
    return self.value

class _IntrinsicDoc(Node):
  pass

class IntrinsicDoc(_IntrinsicDoc):
  def __init__(self, ast):
    super().__init__(ast)
    self.chars = [tr_char(x) for x in ast.chars]
  def code(self):
    return '{' + ''.join(c.code() for c in self.chars) + '}'

class IntrinsicDittoDoc(_IntrinsicDoc):
  def code(self):
    return '{"}'

class Parameter(Node):
  def __init__(self, ast):
    super().__init__(ast)
    self.name = tr_identifier(ast.name)
    self.value = tr_expression(ast.value) if ast.value else None
  def code(self):
    return '{name}{value}'.format(name=self.name.code(), value=(' := ' + self.value.code()) if self.value else '')

class Category(Node):
  pass

class AnyCategory(Category):
  def code(self):
    return '.'

class SetCategory(Category):
  def __init__(self, ast):
    super().__init__(ast)
    self.cat = tr_category(ast.cat)
  def code(self):
    return '{{{cat}}}'.format(cat=(self.cat.code()) if self.cat else '')

class SequenceCategory(Category):
  def __init__(self, ast):
    super().__init__(ast)
    self.cat = tr_category(ast.cat)
  def code(self):
    return '[{cat}]'.format(cat=(self.cat.code()) if self.cat else '')

class IdentCategory(Category):
  def __init__(self, ast):
    super().__init__(ast)
    self.name = tr_identifier(ast.name)
    self.cats = [tr_category(x) for x in ast.cats] if ast.cats else []
  def code(self):
    return '{name}{cats}'.format(name=self.name.code(), cats=('[' + ', '.join(cat.code() for cat in self.cats) + ']') if self.cats else '')

class Space(Node):
  pass

class Whitespace(Space):
  def __init__(self, ast):
    super().__init__(ast)
    self.value = ast.space or ''

class Comment(Space):
  pass

class LineComment(Space):
  def __init__(self, ast):
    super().__init__(ast)
    self.text = ast.text
  def code(self):
    return '//' + self.text

class DocComment(LineComment):
  def __init__(self, ast):
    super().__init__(ast)
    self.command = tr_doc_command(self.text[1:])

class DocCommand:
  def __init__(self, cmd):
    self.command = cmd

class SectionDocCommand(DocCommand):
  def __init__(self, cmd):
    super().__init__(cmd)
    self.level = 0
    while cmd.startswith(SEC_CHAR):
      self.level += 1
      cmd = cmd[1:]
    self.name = cmd.strip()

class TextDocCommand(DocCommand):
  def __init__(self, cmd):
    super().__init__(cmd)
    if self.command.startswith(TEXT_CHAR):
      self.text = self.command[1:].rstrip()
    elif self.command == '':
      self.text = ''
    else:
      assert False

class DittoDocCommand(DocCommand):
  pass

class HideDocCommand(DocCommand):
  pass

class TocDocCommand(DocCommand):
  def rows(self):
    def go(sec, depth):
      for s in sec.children:
        yield (s, depth)
        yield from go(s, depth+1)
    return list(go(self.section, 0))

class Expression(Node):
  pass

def tr_intrinsic_doc(x):
  y = IntrinsicDoc(x)
  if len(y.chars) == 1 and isinstance(y.chars[0], Char) and y.chars[0].value == '"':
    return IntrinsicDittoDoc(x)
  return y

def tr_char(x):
  if x.type == 'char':
    return Char(x)
  elif x.type == 'escape':
    return EscapedChar(x)
  else:
    assert False

def tr_statement(x):
  if x.parseinfo.rule == 'intrinsic_stmt':
    return IntrinsicStatement(x)
  elif x.parseinfo.rule == 'line_comment':
    return tr_line_comment(x)
  elif x.parseinfo.rule == 'block_comment':
    return Comment(x)
  elif x.parseinfo.rule == 'whitespace':
    return Whitespace(x)
  else:
    return Statement(x)

def tr_identifier(x):
  return Identifier(x)

def tr_intrinsic_argument(x):
  return IntrinsicArgument(x)

def tr_parameter(x):
  return Parameter(x)

def tr_category(x):
  if x is None:
    return AnyCategory(x)
  elif x.type == 'any':
    return AnyCategory(x)
  elif x.type == 'seq':
    return SequenceCategory(x)
  elif x.type == 'set':
    return SetCategory(x)
  elif x.type == 'ident':
    return IdentCategory(x)
  else:
    return Category(x)

def tr_scategory(x):
  if x is None:
    return AnyCategory(x)
  elif x.type == 'any':
    return AnyCategory(x)
  elif x.type == 'seq':
    return SequenceCategory(x)
  elif x.type == 'set':
    return SetCategory(x)
  elif x.type == 'ident':
    return IdentCategory(x)
  else:
    return Category(x)

def tr_line_comment(x):
  if x.text.startswith(DOC_CHAR + NOT_DOC_CHAR):
    return LineComment(x)
  elif x.text.startswith(DOC_CHAR):
    return DocComment(x)
  else:
    return LineComment(x)

def tr_expression(x):
  return Expression(x)

def tr_doc_command(cmd):
  if cmd.startswith(SEC_CHAR):
    return SectionDocCommand(cmd)
  elif cmd.startswith(TEXT_CHAR) or cmd=='':
    return TextDocCommand(cmd)
  elif cmd=='ditto':
    return DittoDocCommand(cmd)
  elif cmd=='hide':
    return HideDocCommand(cmd)
  elif cmd=='toc':
    return TocDocCommand(cmd)
  else:
    return DocCommand(cmd)

class Section:
  def __init__(self, name=None, parent=None):
    self.name = name
    self.parent = parent
    self.level = self.parent.level+1 if self.parent else 0
    self.children = []
    self.docs = []
  def get_level(self, level):
    assert level >= 0
    assert level <= self.level
    x = self
    while level < x.level:
      x = x.parent
    return x
  def find_child(self, name):
    for c in self.children:
      if c.name == name:
        return c
  def new_child(self, name):
    c = Section(name=name, parent=self)
    self.children.append(c)
    return c
  def parents(self):
    yield self
    if self.parent:
      yield from self.parent.parents()
  def name_url(self):
    return '#' + self.name.lower().replace(' ', '-')

class IntrinsicGroup:
  def __init__(self):
    self.intrinsics = []

if __name__ == '__main__':
  from pathlib import Path
  from argparse import ArgumentParser

  # parse arguments
  parser = ArgumentParser(description='Produce documentation from MAGMA code.')
  parser.add_argument('filenames', metavar='FILE', type=str, nargs='+', help='MAGMA or markdown source file containing documentation')
  group = parser.add_argument_group('Output')
  group.add_argument('--output-dir', '-o', metavar='DIR', default='.', type=str, help='documentation will be output to this directory')
  group.add_argument('--show-any', action='store_const', const=True, default=False, help='always show the Any type')
  group = parser.add_argument_group('Templates')
  group.add_argument('--arg-tmpl', metavar='TMPL', default='{ref:?~}{name}{cat:? :: }{cat}', help='function argument (ref, name, cat)')
  group.add_argument('--intr-tmpl', metavar='TMPL', default='{groups:~> \n}{{:.intrinsic}}\n\n{doc}', help='intrinsic (groups, doc)')
  group.add_argument('--igroup-tmpl', metavar='TMPL', default='{decls:~> \n}{ret:?> \n}{ret}', help='intrinsic group (decls, ret)')
  group.add_argument('--idecl-tmpl', metavar='TMPL', default='> **{name}** ({args:~, })\n', help='intrinsic declaration (name, args)')
  group.add_argument('--iret-tmpl', metavar='TMPL', default='> -> {cats:~, }\n> {{:.ret}}\n', help='intrinsic return types (cats)')
  group.add_argument('--anycat-tmpl', metavar='TMPL', default='Any', help='Any type')
  group.add_argument('--setcat-tmpl', metavar='TMPL', default='{{{cat}}}', help='SetEnum type (cat)')
  group.add_argument('--seqcat-tmpl', metavar='TMPL', default='[{cat}]', help='SeqEnum type (cat)')
  group.add_argument('--cat-tmpl', metavar='TMPL', default='*{name}*{cats:?[}{cats:~, }{cats:?]}', help='normal types (name, cat)')
  group.add_argument('--section-tmpl', metavar='TMPL', default='{level:*#} {name}\n\n{doc}', help='sections (level, name)')
  group.add_argument('--doc-tmpl', metavar='TMPL', default='{doc}{doc:?\n\n}', help='documentation text (doc)')
  group.add_argument('--toc-tmpl', metavar='TMPL', default='**Contents**\n{rows:~\n}', help='table of contents (rows)')
  group.add_argument('--tocrow-tmpl', metavar='TMPL', default='{level:*  }* [{name}]({url})', help='row of table of contents (name, url, level)')
  args = parser.parse_args()

  # create the output directory
  odir = Path(args.output_dir)
  if not odir.exists():
    odir.mkdir(parents=True)

  # parse source files
  parser = MagmaParser()
  xs = []
  for filename in args.filenames:
    path = Path(filename)
    print('reading', path, '...')
    code = path.open().read()
    if path.suffix in MAGMA_EXTNS:
      print('parsing ...')
      asts = parser.parse(code)
      for ast in asts:
        x = tr_statement(ast)
        x.src_filename = filename
        xs.append(x)
    elif path.suffix in MARKDOWN_EXTNS:
      for line in path.open():
        xs.append(tr_doc_command(line))
    else:
      ValueError(path, 'unknown extension')

  # we now transform the nodes in various ways
  print('transforming ...')

  # replace doc comments by their doc command
  xs = [x.command if isinstance(x, DocComment) else x for x in xs]

  # remember the ordering
  for i,x in enumerate(xs):
    x.src_order = i

  # do the documentation commands
  newxs = []
  root_sec = cur_sec = Section()
  cur_docs = []
  cur_intrs = None
  doc_for = None
  ditto = False
  hide = False
  for x in xs:
    if isinstance(x, SectionDocCommand):
      par_sec = cur_sec.get_level(x.level - 1)
      cur_sec = par_sec.find_child(x.name)
      if cur_sec is None:
        cur_sec = par_sec.new_child(x.name)
        cur_sec.src_order = x.src_order
        newxs.append(cur_sec)
      doc_for = cur_sec
      ditto = False
      hide = False
      cur_intrs = None
    elif isinstance(x, TextDocCommand):
      if doc_for:
        doc_for.docs.append(x)
      else:
        cur_docs.append(x)
    elif isinstance(x, TocDocCommand):
      assert doc_for
      x.section = cur_sec
      doc_for.docs.append(x)
    elif isinstance(x, DittoDocCommand):
      ditto = True
    elif isinstance(x, HideDocCommand):
      hide = True
    elif isinstance(x, DocCommand):
      assert False
    elif isinstance(x, Space):
      if isinstance(x, Whitespace) and max(sum(c=='\n' for c in x.value), sum(c=='\r' for c in x.value)) <= 1:
        pass
      else:
        doc_for = None
    elif isinstance(x, Statement):
      if hide:
        hide = False
        continue
      x.docs = cur_docs
      x.section = cur_sec
      cur_docs = []
      if isinstance(x, IntrinsicStatement):
        # if there are no docs for this, then use the intrinsic's {} docstring
        if not x.docs:
          if isinstance(x.doc, IntrinsicDoc):
            x.docs = [TextDocCommand(TEXT_CHAR + ''.join(c.code() for c in x.doc.chars))]
          elif isinstance(x.doc, IntrinsicDittoDoc):
            ditto = True
          else:
            assert False
        if not ditto:
          cur_intrs = IntrinsicGroup()
          cur_intrs.section = x.section
          cur_intrs.src_order = x.src_order
          newxs.append(cur_intrs)
        cur_intrs.intrinsics.append(x)
        ditto = False
      else:
        newxs.append(x)
      if ditto: print('WARNING: ignoring ditto')
      ditto = False
    else:
      assert False
  xs = newxs

  # filter
  newxs = []
  for x in xs:
    if isinstance(x, Section):
      newxs.append(x)
    elif isinstance(x, IntrinsicStatement):
      assert False
    elif isinstance(x, IntrinsicGroup):
      newxs.append(x)
    else:
      if hasattr(x, 'docs') and x.docs:
        print('WARNING: ignoring docs on', x)
  xs = newxs

  # sort
  xs.sort(key = lambda x: (x.src_order, -1) if isinstance(x,Section) else (x.section.src_order, x.src_order))

  # output
  section_files_depth = 1
  ENCODE_REPLACE = {'*': '\\*', '\'': '\\\''}
  PATH_REPLACE = {' ': '-'}
  def replace_many(x, rs):
    return ''.join(rs.get(c, c) for c in x)
  def path_replace(x):
    return replace_many(x, PATH_REPLACE)
  def section_to_path(sec):
    return '-'.join(reversed(list(path_replace(s.name.lower()) for s in sec.parents() if s.name))) + '.md'
  def out_encode(x):
    return replace_many(x, ENCODE_REPLACE)
  def cat_out(x, can_hide_any=False):
    if isinstance(x, AnyCategory):
      return template_subs(args.anycat_tmpl) if (args.show_any or not can_hide_any) else ''
    elif isinstance(x, SetCategory):
      return template_subs(args.setcat_tmpl, cat=cat_out(x.cat))
    elif isinstance(x, SequenceCategory):
      return template_subs(args.seqcat_tmpl, cat=cat_out(x.cat))
    elif isinstance(x, IdentCategory):
      return template_subs(args.cat_tmpl, name=out_encode(x.name.code()), cats=[cat_out(c) for c in x.cats] if x.cats else [])
    else:
      assert False
  def iarg_out(x):
    return template_subs(args.arg_tmpl, ref=x.is_ref, name=out_encode(x.name.code()), cat=cat_out(x.cat, can_hide_any=True))
  def doc_out(text):
    return template_subs(args.doc_tmpl, doc=text)
  def intr_out(x):
    # group the intrinsics by their return values
    gs = dict()
    for intr in x.intrinsics:
      r = template_subs(args.iret_tmpl, cats=[cat_out(r) for r in intr.returns]) if intr.returns else ''
      if r in gs:
        gs[r].append(intr)
      else:
        gs[r] = [intr]
    gs = sorted(gs.items(), key = lambda z: min(intr.src_order for intr in z[1]))
    # apply the template
    return template_subs(args.intr_tmpl, groups=[template_subs(args.igroup_tmpl, decls=[template_subs(args.idecl_tmpl, name=out_encode(intr.name.code()), args=[iarg_out(arg) for arg in intr.args]) for intr in intrs], ret=ret) for ret,intrs in gs], doc=doc_out('\n\n'.join('\n'.join(d.text for d in intr.docs) for ret,intrs in gs for intr in intrs)))
  def toc_out(x):
    return template_subs(args.toc_tmpl, rows=[template_subs(args.tocrow_tmpl, name=sec.name, url=sec.name_url(), level=lvl) for sec,lvl in x.rows()])
  def doctext_out(x):
    if isinstance(x, TocDocCommand):
      return toc_out(x)
    else:
      return x.text
  def section_out(x):
    return template_subs(args.section_tmpl, level=x.level, name=x.name, doc=doc_out('\n'.join(doctext_out(d) for d in x.docs)))
  for x in xs:
    if isinstance(x, Section):
      if x.level <= section_files_depth:
        opath = odir.joinpath(section_to_path(x))
        if not opath.parent.exists():
          opath.parent.mkdir(parents=True)
        ofile = opath.open('wt')
        print('writing', opath, '...')
      ofile.write(section_out(x))
    elif isinstance(x, IntrinsicStatement):
      assert False
    elif isinstance(x, IntrinsicGroup):
      ofile.write(intr_out(x))
