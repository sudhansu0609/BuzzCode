//! tree-sitter parsing → symbols + reference counts.

use crate::lang::Language;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use std::cell::RefCell;
use std::collections::HashMap;
use tree_sitter::{Node, Parser};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SymKind { Fn, Method, Struct, Enum, Trait, Class, Interface, Const, Static, Type, Module, Impl, Macro }

impl SymKind {
    pub fn label(self) -> &'static str {
        match self { SymKind::Fn => "fn", SymKind::Method => "method", SymKind::Struct => "struct", SymKind::Enum => "enum", SymKind::Trait => "trait", SymKind::Class => "class",
            SymKind::Interface => "interface", SymKind::Const => "const", SymKind::Static => "static", SymKind::Type => "type", SymKind::Module => "mod", SymKind::Impl => "impl", SymKind::Macro => "macro" }
    }
    pub fn parse(s: &str) -> Option<SymKind> {
        Some(match s.to_ascii_lowercase().as_str() {
            "fn" | "function" | "func" => SymKind::Fn, "method" => SymKind::Method, "struct" => SymKind::Struct, "enum" => SymKind::Enum, "trait" => SymKind::Trait,
            "class" => SymKind::Class, "interface" => SymKind::Interface, "const" => SymKind::Const, "static" | "var" => SymKind::Static, "type" => SymKind::Type,
            "mod" | "module" | "namespace" => SymKind::Module, "impl" => SymKind::Impl, "macro" => SymKind::Macro, _ => return None,
        })
    }
    /// Weight of a reference to this kind of symbol when building the graph.
    pub fn weight(self) -> f32 {
        match self { SymKind::Fn | SymKind::Method => 1.0, SymKind::Struct | SymKind::Class | SymKind::Enum | SymKind::Trait | SymKind::Interface | SymKind::Type => 1.2, SymKind::Const | SymKind::Static => 0.6, SymKind::Module | SymKind::Impl | SymKind::Macro => 0.5 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: SmolStr,
    pub kind: SymKind,
    /// 1-based inclusive lines.
    pub line_start: u32,
    pub line_end: u32,
    /// First line of the definition, trimmed, ≤ 120 chars.
    pub signature: String,
    /// Index of the enclosing symbol in the same file, if any.
    pub parent: Option<u32>,
    pub is_public: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ParsedFile {
    pub symbols: Vec<Symbol>,
    /// identifier → occurrence count (references for the graph).
    pub refs: HashMap<SmolStr, u16>,
}

thread_local! {
    static PARSERS: RefCell<HashMap<Language, Parser>> = RefCell::new(HashMap::new());
}

pub fn parse(lang: Language, src: &[u8]) -> ParsedFile {
    PARSERS.with(|cell| {
        let mut map = cell.borrow_mut();
        let parser = map.entry(lang).or_insert_with(|| { let mut p = Parser::new(); p.set_language(&lang.ts_language()).expect("grammar"); p });
        let Some(tree) = parser.parse(src, None) else { return ParsedFile::default() };
        let mut out = ParsedFile::default();
        let defs = lang.def_kinds();
        let refk = lang.ref_kinds();
        let max_symbols = if lang.is_code() { 4000 } else { 200 };
        let max_depth = if lang.is_code() { usize::MAX } else { 3 }; // json/toml: top-level keys only
        let mut stack: Vec<(Node, Option<u32>, usize)> = vec![(tree.root_node(), None, 0)];
        // Iterative DFS preserving parent symbol index.
        while let Some((node, parent, depth)) = stack.pop() {
            let kind = node.kind();
            let mut my_parent = parent;
            if depth > max_depth && !lang.is_code() { continue; }
            if out.symbols.len() < max_symbols { if let Some((_, sk)) = defs.iter().find(|(k, _)| *k == kind) {
                if let Some(name) = symbol_name(lang, node, src) {
                    if !name.is_empty() && name.len() < 200 {
                        let start = node.start_position().row as u32 + 1;
                        let end = node.end_position().row as u32 + 1;
                        let sig = signature_line(node, src);
                        let is_public = is_public(lang, node, &name, &sig);
                        out.symbols.push(Symbol { name: SmolStr::new(&name), kind: *sk, line_start: start, line_end: end, signature: sig, parent, is_public });
                        my_parent = Some((out.symbols.len() - 1) as u32);
                    }
                }
            } }
            if node.child_count() == 0 {
                if refk.contains(&kind) {
                    if let Ok(t) = node.utf8_text(src) {
                        if t.len() > 1 && t.len() < 100 && !t.chars().all(|c| c.is_ascii_digit()) {
                            *out.refs.entry(SmolStr::new(t)).or_default() += 1;
                        }
                    }
                }
                continue;
            }
            let mut cursor = node.walk();
            let children: Vec<Node> = node.children(&mut cursor).collect();
            for c in children.into_iter().rev() { stack.push((c, my_parent, depth + 1)); }
        }
        // Symbols are pushed in DFS pre-order already; sort by line for stable output.
        out.symbols.sort_by_key(|s| (s.line_start, s.line_end));
        out
    })
}

fn symbol_name(lang: Language, node: Node, src: &[u8]) -> Option<String> {
    let text = |n: Node| n.utf8_text(src).ok().map(|s| s.trim().to_string());
    match lang {
        Language::Markdown => {
            // heading: text after the '#'s
            let t = text(node)?;
            Some(t.trim_start_matches('#').trim().trim_end_matches('=').trim_end_matches('-').trim().to_string())
        }
        Language::Json | Language::Toml => {
            if node.kind() == "pair" {
                // only top-level-ish pairs are interesting; keep key text
                let key = node.child_by_field_name("key").or_else(|| node.child(0))?;
                return text(key).map(|s| s.trim_matches('"').to_string());
            }
            let n = node.child_by_field_name("name").or_else(|| node.named_child(0))?;
            text(n)
        }
        _ => {
            if let Some(n) = node.child_by_field_name("name") { return text(n); }
            if let Some(n) = node.child_by_field_name("declarator") { return declarator_name(n, src); }
            if node.kind() == "impl_item" {
                // impl Trait for Type → "Type" (or impl Type)
                let ty = node.child_by_field_name("type")?;
                let tr = node.child_by_field_name("trait").and_then(|t| text(t));
                let t = text(ty)?;
                return Some(match tr { Some(tr) => format!("{tr} for {t}"), None => t });
            }
            if matches!(node.kind(), "lexical_declaration" | "variable_declaration" | "declaration") {
                // const foo = ...; int foo(...) — find first declarator's name
                let mut cursor = node.walk();
                for c in node.children(&mut cursor) {
                    if c.kind().ends_with("declarator") {
                        if let Some(n) = c.child_by_field_name("name").or_else(|| c.child_by_field_name("declarator")) {
                            return declarator_name(n, src);
                        }
                    }
                }
                return None;
            }
            if node.kind() == "template_declaration" { return None; }
            // fallback: first identifier-like child
            let mut cursor = node.walk();
            for c in node.children(&mut cursor) {
                if c.kind().contains("identifier") { return text(c); }
            }
            None
        }
    }
}

fn declarator_name(n: Node, src: &[u8]) -> Option<String> {
    let mut cur = n;
    loop {
        if cur.kind().contains("identifier") { return cur.utf8_text(src).ok().map(str::to_string); }
        let next = cur.child_by_field_name("declarator").or_else(|| cur.child_by_field_name("name")).or_else(|| {
            let mut c = cur.walk();
            cur.children(&mut c).find(|x| x.kind().contains("identifier") || x.kind().ends_with("declarator"))
        })?;
        cur = next;
    }
}

fn signature_line(node: Node, src: &[u8]) -> String {
    let start = node.start_byte();
    let end = node.end_byte().min(start + 400);
    let s = String::from_utf8_lossy(&src[start..end]);
    let first = s.lines().next().unwrap_or("").trim();
    let mut sig: String = first.chars().take(120).collect();
    if let Some(i) = sig.find('{') { sig.truncate(i); }
    sig.trim_end().trim_end_matches(':').to_string()
}

fn is_public(lang: Language, node: Node, name: &str, sig: &str) -> bool {
    match lang {
        Language::Rust => sig.starts_with("pub"),
        Language::Go => name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false),
        Language::Python => !name.starts_with('_'),
        Language::TypeScript | Language::Tsx | Language::JavaScript => sig.contains("export") || node.parent().map(|p| p.kind() == "export_statement").unwrap_or(false),
        Language::Java | Language::Cpp | Language::C => sig.contains("public") || lang == Language::C,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rust_symbols() {
        let src = b"pub struct Foo { x: i32 }\nimpl Foo {\n    pub fn new() -> Self { Foo { x: bar() } }\n}\nfn bar() -> i32 { 1 }\n";
        let p = parse(Language::Rust, src);
        let names: Vec<&str> = p.symbols.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["Foo", "Foo", "new", "bar"]);
        assert_eq!(p.symbols[2].parent, Some(1));
        assert!(p.refs.contains_key("bar"));
        assert!(p.symbols[0].is_public);
    }
    #[test]
    fn python_symbols() {
        let p = parse(Language::Python, b"class A:\n    def m(self):\n        return helper()\n\ndef helper():\n    pass\n");
        let names: Vec<&str> = p.symbols.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["A", "m", "helper"]);
    }
    #[test]
    fn ts_symbols() {
        let p = parse(Language::TypeScript, b"export function f(a: number) { return g(a); }\nexport const x = 1;\nclass K { run() {} }\n");
        let names: Vec<&str> = p.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"f") && names.contains(&"x") && names.contains(&"K") && names.contains(&"run"), "{names:?}");
    }
}
