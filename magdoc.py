from magmaparser import MagmaParser
import re
from itertools import groupby
import string

def dedupe(xs):
  ys = []
  for x in xs:
    if x not in ys:
      ys.append(x)
  return ys

class Data:
  pass

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
    self.constructor = 'ext' if (self.name.code()=='ExtConstructor' and len(self.args)==2) else 'elt' if (self.name.code()=='EltConstructor' and len(self.args)==2) else 'quo' if (self.name.code()=='QuoConstructor' and len(self.args)==2) else None
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
  def anchors(self):
    name = self.name.anchor()
    yield name
    if len(self.args) == 0:
      yield '{}--noargs'.format(name)
    elif len(self.args) > 1:
      yield '{}--{}--etc'.format(name, self.args[0].cat.anchor())
    if len(self.args) > 0:
      yield '{}--{}'.format(name, '--'.join(arg.cat.anchor() for arg in self.args))

class Identifier(Node):
  def __init__(self, ast):
    super().__init__(ast)
    self.value = ast.value
  def code(self):
    if re_ident.match(self.value):
      return self.value
    else:
      return '\'' + self.value + '\''
  def anchor(self):
    return self.value

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
  def anchor(self):
    return 'any'

class SetCategory(Category):
  def __init__(self, ast):
    super().__init__(ast)
    self.cat = tr_category(ast.cat)
  def code(self):
    return '{{{cat}}}'.format(cat=(self.cat.code()) if self.cat else '')
  def anchor(self):
    return 'set' + ('' if isinstance(self.cat, AnyCategory) else '-'+self.cat.anchor())

class IndexedSetCategory(Category):
  def __init__(self, ast):
    super().__init__(ast)
    self.cat = tr_category(ast.cat)
  def code(self):
    return '{{@{cat}@}}'.format(cat=(self.cat.code()) if self.cat else '')
  def anchor(self):
    return 'iset' + ('' if isinstance(self.cat, AnyCategory) else '-'+self.cat.anchor())

class MultisetCategory(Category):
  def __init__(self, ast):
    super().__init__(ast)
    self.cat = tr_category(ast.cat)
  def code(self):
    return '{{*{cat}*}}'.format(cat=(self.cat.code()) if self.cat else '')
  def anchor(self):
    return 'mset' + ('' if isinstance(self.cat, AnyCategory) else '-'+self.cat.anchor())

class SequenceCategory(Category):
  def __init__(self, ast):
    super().__init__(ast)
    self.cat = tr_category(ast.cat)
  def code(self):
    return '[{cat}]'.format(cat=(self.cat.code()) if self.cat else '')
  def anchor(self):
    return 'seq' + ('' if isinstance(self.cat, AnyCategory) else '-'+self.cat.anchor())

class IdentCategory(Category):
  def __init__(self, ast):
    super().__init__(ast)
    self.name = tr_identifier(ast.name)
    self.cats = [tr_category(x) for x in ast.cats] if ast.cats else []
  def code(self):
    return '{name}{cats}'.format(name=self.name.code(), cats=('[' + ', '.join(cat.code() for cat in self.cats) + ']') if self.cats else '')
  def anchor(self):
    return self.name.anchor() + ('-'+'-'.join(cat.anchor() for cat in self.cats) if self.cats else '')

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
  def __init__(self, cmd=None):
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

class HideAllDocCommand(DocCommand):
  pass

class HideNoneDocCommand(DocCommand):
  pass

class LabelDocCommand(DocCommand):
  def __init__(self, cmd):
    super().__init__(cmd)
    words = cmd.split(None, 1)
    assert len(words) == 2
    assert words[0] == 'label'
    self.name = words[1]

class TocDocCommand(DocCommand):
  def rows(self):
    def go(sec, depth):
      for s in sec.children:
        yield (s, depth)
        yield from go(s, depth+1)
    return list(go(self.section, 0))

class ParamDocCommand(DocCommand):
  def __init__(self, cmd):
    super().__init__(cmd)
    words = cmd.split(None, 2)
    assert len(words) >= 2
    assert words[0] == 'param'
    cmd = words[2] if len(words) > 2 else ''
    self.hide = False
    self.doc = ''
    if cmd == '/hide':
      self.hide = True
    else:
      self.doc = cmd
    x = words[1].split(':=', 1)
    self.param = x[0]
    self.default = x[1] if len(x) > 1 else ''

class SectionRefDocCommand(DocCommand):
  def __init__(self, cmd):
    super().__init__(cmd)
    words = cmd.split(None, 1)
    assert len(words) == 2
    assert words[0] == 'sec'
    self.name = words[1]

class PriorityDocCommand(DocCommand):
  def __init__(self, cmd):
    super().__init__(cmd)
    words = cmd.split()
    assert len(words) == 2
    assert words[0] == 'priority'
    self.priority = int(words[1])

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
  elif x.type == 'iset':
    return IndexedSetCategory(x)
  elif x.type == 'mset':
    return MultisetCategory(x)
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
  elif x.type == 'iset':
    return IndexedSetCategory(x)
  elif x.type == 'mset':
    return MultisetCategory(x)
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
  elif cmd.startswith('param '):
    return ParamDocCommand(cmd)
  elif cmd.startswith('label '):
    return LabelDocCommand(cmd)
  elif cmd=='hide-all':
    return HideAllDocCommand(cmd)
  elif cmd=='hide-none':
    return HideNoneDocCommand(cmd)
  elif cmd.startswith('sec '):
    return SectionRefDocCommand(cmd)
  elif cmd.startswith('priority '):
    return PriorityDocCommand(cmd)
  else:
    raise ValueError('unknown doc command', cmd)

class Section:
  def __init__(self, name=None, parent=None, src_order=None):
    self.name = name
    self.parent = parent
    self.level = self.parent.level+1 if self.parent else 0
    self.children = []
    self.docs = []
    self.src_order = src_order
    self.nodes = []
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
  def new_child(self, name, **kwargs):
    c = Section(name=name, parent=self, **kwargs)
    self.children.append(c)
    return c
  def parents(self):
    yield self
    if self.parent:
      yield from self.parent.parents()
  def anchor(self):
    return ''.join(c if c in 'abcdefghijklmnopqrstuvwxyz-_0123456789' else '-' if c in ' ' else '' for c in self.name.lower())
  def prioritize_nodes(self, key):
    self.nodes = sorted(self.nodes, key=key)
  def prioritize(self, key, recurse=True, nodes=True):
    self.children = sorted(self.children, key=key)
    if nodes:
      self.prioritize_nodes(key)
    if recurse:
      for x in self.children:
        x.prioritize(key, nodes=nodes)

class IntrinsicGroup:
  def __init__(self):
    self.intrinsics = []
  def anchors(self):
    for intr in self.intrinsics:
      yield from intr.anchors()

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
  group.add_argument('--intr-tmpl', metavar='TMPL', default='{anchors:?<a id="}{anchors:~"></a><a id="}{anchors:?"></a>\n}{groups:~> \n}{{:.intrinsic}}\n\n{doc}{params:?**Parameters**}\n{params:~\n}{params:?\n\n}', help='intrinsic (groups, doc, params, anchors)')
  group.add_argument('--igroup-tmpl', metavar='TMPL', default='{decls:~> \n}{ret:?> \n}{ret}', help='intrinsic group (decls, ret)')
  group.add_argument('--idecl-tmpl', metavar='TMPL', default='> **{name}** ({args:~, })\n', help='intrinsic declaration (name, args)')
  group.add_argument('--cdecl-tmpl', metavar='TMPL', default='> **{name}** \\<{content}>\n', help='constructor declaration (name, content)')
  group.add_argument('--extcon-tmpl', metavar='TMPL', default='{lhs} \\| {rhs}', help='ext constructor content declaration (lhs, rhs)')
  group.add_argument('--eltcon-tmpl', metavar='TMPL', default='{lhs} \\| {rhs}', help='elt constructor content declaration (lhs, rhs)')
  group.add_argument('--quocon-tmpl', metavar='TMPL', default='{lhs} \\| {rhs}', help='elt constructor content declaration (lhs, rhs)')
  group.add_argument('--iret-tmpl', metavar='TMPL', default='> -> {cats:~, }\n> {{:.ret}}\n', help='intrinsic return types (cats)')
  group.add_argument('--anycat-tmpl', metavar='TMPL', default='Any', help='Any type')
  group.add_argument('--setcat-tmpl', metavar='TMPL', default='{{{cat}}}', help='SetEnum type (cat)')
  group.add_argument('--isetcat-tmpl', metavar='TMPL', default='{{@{cat}@}}', help='SetIndx type (cat)')
  group.add_argument('--msetcat-tmpl', metavar='TMPL', default='{{*{cat}*}}', help='SetMulti type (cat)')
  group.add_argument('--seqcat-tmpl', metavar='TMPL', default='[{cat}]', help='SeqEnum type (cat)')
  group.add_argument('--cat-tmpl', metavar='TMPL', default='*{name}*{cats:?[}{cats:~, }{cats:?]}', help='normal types (name, cat)')
  group.add_argument('--section-tmpl', metavar='TMPL', default='{level:*#} {name}\n{{:#{anchor}}}\n\n{doc}', help='sections (level, name, anchor)')
  group.add_argument('--doc-tmpl', metavar='TMPL', default='{doc}{doc:?\n\n}', help='documentation text (doc)')
  group.add_argument('--toc-tmpl', metavar='TMPL', default='\n**Contents**\n{rows:~\n}', help='table of contents (rows)')
  group.add_argument('--tocrow-tmpl', metavar='TMPL', default='{level:*  }* [{name}](#{anchor})', help='row of table of contents (name, anchor, level)')
  group.add_argument('--param-tmpl', metavar='TMPL', default='- `{name}{default:? := }{default}`{doc:?: }{doc}', help='a function parameter (name, default, doc)')
  group.add_argument('--docarg-tmpl', metavar='TMPL', default='`{name}`', help='the name of an argument in a doc string (name)')
  group.add_argument('--docparam-tmpl', metavar='TMPL', default='`{name}`', help='the name of a parameter in a doc string (name)')
  group.add_argument('--docprotect-rx', metavar='REGEX', default=r'```.+?```|``.+?``|`.+?`|\$\$.*?\$\$|\$.+?\$', help='fragments of documentation matching this will not be altered')
  group.add_argument('--docident-rx', metavar='REGEX', default=r'\b{name}(?=(th|st)?\b)', help='regular expression for an identifier in a doc string, used to highlight argument and parameter names automatically')
  group.add_argument('--anchor-tmpl', metavar='TMPL', default='<a id="{name}"></a>', help='anchor (name)')
  args = parser.parse_args()

  # create the output directory
  odir = Path(args.output_dir)
  if not odir.exists():
    odir.mkdir(parents=True)

  # parse source files
  parser = MagmaParser()
  xs = []
  for filename in args.filenames:
    if filename.startswith('//'+DOC_CHAR):
      xs.append(tr_doc_command(filename[3:]))
    else:
      path = Path(filename)
      print('parsing', path, '...')
      code = path.open().read()
      if path.suffix in MAGMA_EXTNS:
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

  # perform doc commands which act "globally" (such as sectioning) and attach the rest to their targets
  newxs = []
  root_sec = cur_sec = Section(src_order=-1)
  docs_for_next = []
  docs_for_sec = False
  hide_all = False
  sec_labels = dict()
  for x in xs:
    if isinstance(x, HideNoneDocCommand):
      hide_all = False
    elif hide_all:
      pass
    elif isinstance(x, HideAllDocCommand):
      hide_all = True
    elif isinstance(x, SectionDocCommand):
      par_sec = cur_sec.get_level(x.level - 1)
      cur_sec = par_sec.find_child(x.name)
      if cur_sec is None:
        cur_sec = par_sec.new_child(x.name, src_order=x.src_order)
        newxs.append(cur_sec)
      assert cur_sec is not None
      docs_for_sec = True
      cur_sec.docs += docs_for_next
      docs_for_next = []
    elif isinstance(x, SectionRefDocCommand):
      cur_sec = sec_labels[x.name]
      docs_for_sec = False
      cur_sec.docs += docs_for_next
      docs_for_next = []
    elif isinstance(x, LabelDocCommand):
      if docs_for_sec:
        sec_labels[x.name] = cur_sec
      else:
        print('WARNING: ignoring label doc command')
    elif isinstance(x, DocCommand):
      if docs_for_sec:
        cur_sec.docs.append(x)
      else:
        docs_for_next.append(x)
    elif isinstance(x, Space):
      if docs_for_sec and isinstance(x, Whitespace):
        if sum(c=='\n' for c in x.value)>1 or sum(c=='\r' for c in x.value)>1 or sum(c!='\n' and c!='\t' for c in x.value)>0:
          docs_for_sec = False
    elif isinstance(x, Statement):
      docs_for_sec = False
      x.section = cur_sec
      x.docs = getattr(x, 'docs', []) + docs_for_next
      docs_for_next = []
      newxs.append(x)
    else:
      assert False
  xs = newxs

  # # if an intrinsic has no doc commands, get one from the "{...}" docstring
  # for x in xs:
  #   if isinstance(x, IntrinsicStatement) and not x.docs:
  #     if isinstance(x.doc, IntrinsicDoc):
  #       x.docs = [TextDocCommand(TEXT_CHAR + ''.join(c.code() for c in x.doc.chars))]
  #     elif isinstance(x.doc, IntrinsicDittoDoc):
  #       x.docs = [DittoDocCommand()]
  #     else:
  #       assert False

  # apply doc commands attached to nodes
  newxs = []
  cur_intrs = None
  for x in xs:
    docs = x.docs
    x.docs = []
    ditto = False
    hide = False
    for d in docs:
      if isinstance(d, TextDocCommand):
        x.docs.append(d)
      elif isinstance(d, TocDocCommand):
        if isinstance(x, Section):
          d.section = x
          x.docs.append(d)
        else:
          print('WARNING: ignoring toc doc command: not a section')
      elif isinstance(d, DittoDocCommand):
        ditto = True
      elif isinstance(d, HideDocCommand):
        hide = True
      elif isinstance(d, ParamDocCommand):
        if isinstance(x, IntrinsicStatement):
          for p in x.params:
            if p.name.code() == d.param:
              p.hide = getattr(p, 'hide', False) or d.hide
              if getattr(p, 'value_doc', ''):
                print('WARNING: ignoring param doc command default: already set')
              else:
                p.value_doc = d.default
              if getattr(p, 'doc_text', ''):
                p.doc_text += ' ' + d.doc
              else:
                p.doc_text = d.doc
              break
          else:
            print('WARNING: ignoring param doc command: invalid parameter name')
        else:
          print('WARNING: ignoring param doc command: not an intrinsic')
      elif isinstance(d, PriorityDocCommand):
        x.priority = d.priority
      else:
        assert isinstance(DocCommand)
        assert False
    # if an intrinsic has no doc yet, use its docstring
    if isinstance(x, IntrinsicStatement) and not x.docs and not ditto:
      if isinstance(x.doc, IntrinsicDoc):
        x.docs.append(TextDocCommand(TEXT_CHAR + ''.join(c.code() for c in x.doc.chars)))
      elif isinstance(x.doc, IntrinsicDittoDoc):
        ditto = True
      else:
        assert False
    # hide
    if hide:
      continue
    # ditto
    if isinstance(x, IntrinsicStatement):
      if not ditto:
        cur_intrs = IntrinsicGroup()
        cur_intrs.section = x.section
        cur_intrs.src_order = x.src_order
        newxs.append(cur_intrs)
      cur_intrs.intrinsics.append(x)
      # the priority of an intrinsic group is the maximum assigned priority of its intrinsics
      if hasattr(cur_intrs, 'priority'):
        if hasattr(x, 'priority'):
          cur_intrs.priority = max(cur_intrs.priority, x.priority)
      elif hasattr(x, 'priority'):
        cur_intrs.priority = x.priority
    else:
      if ditto: print('WARNING: ignoring ditto')
      newxs.append(x)
  xs = newxs

  # highlight argument and parameter names
  re_protect = re.compile(r'(.*?)($|' + args.docprotect_rx + r')')
  for x in xs:
    if isinstance(x, IntrinsicGroup):
      for i in x.intrinsics:
        if i.args or i.params:
          regex = args.docident_rx.replace(r'{name}', r'((?P<arg>' + r'|'.join(re.escape(a.name.code()) for a in i.args) + r')|(?P<param>' +  r'|'.join(re.escape(p.name.code()) for p in i.params) + r'))')
          r = re.compile(regex)
          def repl(m):
            if m.group('arg'):
              return template_subs(args.docarg_tmpl, name=m.group('arg'))
            elif m.group('param'):
              return template_subs(args.docparam_tmpl, name=m.group('param'))
          def repl0(m):
            return r.sub(repl, m.group(1)) + m.group(2)
          for d in i.docs:
            if isinstance(d, TextDocCommand):
              d.text = re_protect.sub(repl0, d.text)

  # attach nodes to sections
  for x in xs:
    if not isinstance(x, Section):
      x.section.nodes.append(x)

  # order the sections and nodes, first by priority, then by source order
  root_sec.prioritize(lambda x: (-getattr(x,'priority',0), x.src_order))

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
  def ident_out(x):
    return out_encode(x.code())
  def cat_out(x, can_hide_any=False):
    if isinstance(x, AnyCategory):
      return template_subs(args.anycat_tmpl) if (args.show_any or not can_hide_any) else ''
    elif isinstance(x, SetCategory):
      return template_subs(args.setcat_tmpl, cat=cat_out(x.cat, can_hide_any=True))
    elif isinstance(x, SequenceCategory):
      return template_subs(args.seqcat_tmpl, cat=cat_out(x.cat, can_hide_any=True))
    elif isinstance(x, IndexedSetCategory):
      return template_subs(args.isetcat_tmpl, cat=cat_out(x.cat, can_hide_any=True))
    elif isinstance(x, MultisetCategory):
      return template_subs(args.msetcat_tmpl, cat=cat_out(x.cat, can_hide_any=True))
    elif isinstance(x, IdentCategory):
      return template_subs(args.cat_tmpl, name=out_encode(x.name.code()), cats=[cat_out(c) for c in x.cats] if x.cats else [])
    else:
      assert False
  def iarg_out(x):
    return template_subs(args.arg_tmpl, ref=x.is_ref, name=out_encode(x.name.code()), cat=cat_out(x.cat, can_hide_any=True))
  def doc_out(text):
    return template_subs(args.doc_tmpl, doc=text)
  def intr_decl_out(x):
    if x.constructor:
      if x.constructor == 'ext':
        content = template_subs(args.extcon_tmpl, lhs=iarg_out(x.args[0]), rhs='...')
      elif x.constructor == 'elt':
        content = template_subs(args.eltcon_tmpl, lhs=iarg_out(x.args[0]), rhs='...')
      elif x.constructor == 'quo':
        content = template_subs(args.quocon_tmpl, lhs=iarg_out(x.args[0]), rhs='...')
      else:
        assert False
      return template_subs(args.cdecl_tmpl, name=x.constructor, content=content)
    else:
      return template_subs(args.idecl_tmpl, name=out_encode(x.name.code()), args=[iarg_out(arg) for arg in x.args])
  def toc_out(x):
    return template_subs(args.toc_tmpl, rows=[template_subs(args.tocrow_tmpl, name=sec.name, anchor=sec.fixed_anchor, level=lvl) for sec,lvl in x.rows()])
  def doctext_out(x):
    if isinstance(x, TocDocCommand):
      return toc_out(x)
    else:
      return x.text
  class Walker:
    def __init__(self, root_sec):
      self.walk(root_sec)
    def set_opath(self, opath):
      self.opath = opath
      if not opath.parent.exists():
        opath.parent.mkdir(parents=True)
      self.ofile = opath.open('wt')
      print('writing', opath, '...')
      self.anchor_idxs = dict()
    def next_anchor_idx(self, name):
      n = self.anchor_idxs.get(name, 0) + 1
      self.anchor_idxs[name] = n
      return n
    def next_anchor(self, name):
      n = self.next_anchor_idx(name)
      return name if n==1 else '{}-{}'.format(name, n)
    def write(self, *args, **kwds):
      self.ofile.write(*args, **kwds)
    def walk(self, sec):
      if sec.level > 0:
        if sec.level <= section_files_depth:
          self.set_opath(odir.joinpath(section_to_path(sec)))
          self.fix_anchors(sec)
        self.write(self.section_out(sec))
        for x in sec.nodes:
          if isinstance(x, IntrinsicGroup):
            self.write(self.intr_out(x))
          elif hasattr(x, 'docs') and x.docs:
            print('ignoring docs on', x)
      for ssec in sec.children:
        self.walk(ssec)
    def fix_anchors(self, sec):
      sec.fixed_anchor = self.next_anchor(sec.anchor())
      for ssec in sec.children:
        self.fix_anchors(ssec)
    def intr_out(self, x):
      # group the intrinsics by their return values
      gs = dict()
      for intr in x.intrinsics:
        r = template_subs(args.iret_tmpl, cats=[cat_out(r) for r in intr.returns]) if intr.returns else ''
        if r in gs:
          gs[r].append(intr)
        else:
          gs[r] = [intr]
      gs = sorted(gs.items(), key = lambda z: min(intr.src_order for intr in z[1]))
      # collate parameters
      params = []
      for ret, intrs in gs:
        for intr in intrs:
          for param in intr.params:
            name = ident_out(param.name)
            dflt = getattr(param, 'value_doc', '')
            doc = getattr(param, 'doc_text', '')
            hide = getattr(param, 'hide', False)
            for i,p in enumerate(params):
              if p[0] == name:
                if dflt and p[1]:
                  print('WARNING: ignoring a parameter default')
                else:
                  dflt = dflt or p[1]
                if doc and p[2]:
                  doc = p[2] + ' ' + doc
                else:
                  doc = doc or p[2]
                hide = hide or p[3]
                params[i] = (name, dflt, doc, hide)
                break
            else:
              params.append((name, dflt, doc, hide))
      # apply the template
      return template_subs(args.intr_tmpl,
        groups=[template_subs(args.igroup_tmpl, decls=[intr_decl_out(intr) for intr in intrs], ret=ret) for ret,intrs in gs],
        doc=doc_out('\n\n'.join('\n'.join(d.text for d in intr.docs) for ret,intrs in gs for intr in intrs)),
        params=[template_subs(args.param_tmpl, name=name, default=dflt, doc=doc) for name,dflt,doc,hide in params if not hide],
        anchors=[self.next_anchor(a) for a in dedupe(x.anchors())]
      )
    def section_out(self, x):
      return template_subs(args.section_tmpl, level=x.level, name=x.name, doc=doc_out('\n'.join(doctext_out(d) for d in x.docs)), anchor=x.fixed_anchor)
  Walker(root_sec)
