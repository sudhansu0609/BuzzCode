//! Supported languages and their tree-sitter grammars + symbol node kinds.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Language { Rust, Python, TypeScript, Tsx, JavaScript, Go, C, Cpp, Java, Json, Toml, Markdown }

impl Language {
    pub fn from_path(p: &Path) -> Option<Language> {
        let ext = p.extension()?.to_str()?.to_ascii_lowercase();
        Some(match ext.as_str() {
            "rs" => Language::Rust,
            "py" | "pyi" => Language::Python,
            "ts" | "mts" | "cts" => Language::TypeScript,
            "tsx" => Language::Tsx,
            "js" | "mjs" | "cjs" | "jsx" => Language::JavaScript,
            "go" => Language::Go,
            "c" | "h" => Language::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Language::Cpp,
            "java" => Language::Java,
            "json" => Language::Json,
            "toml" => Language::Toml,
            "md" | "markdown" => Language::Markdown,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Language::Rust => "rust", Language::Python => "python", Language::TypeScript => "typescript", Language::Tsx => "typescript",
            Language::JavaScript => "javascript", Language::Go => "go", Language::C => "c", Language::Cpp => "cpp", Language::Java => "java",
            Language::Json => "json", Language::Toml => "toml", Language::Markdown => "markdown",
        }
    }

    pub fn ts_language(self) -> tree_sitter::Language {
        match self {
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
            Language::Python => tree_sitter_python::LANGUAGE.into(),
            Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Language::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Language::Go => tree_sitter_go::LANGUAGE.into(),
            Language::C => tree_sitter_c::LANGUAGE.into(),
            Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Language::Java => tree_sitter_java::LANGUAGE.into(),
            Language::Json => tree_sitter_json::LANGUAGE.into(),
            Language::Toml => tree_sitter_toml_ng::LANGUAGE.into(),
            Language::Markdown => tree_sitter_md::LANGUAGE.into(),
        }
    }

    /// Node kinds that define symbols → (kind, SymKind). Name is taken from the `name` field
    /// (or the first identifier-ish child).
    pub fn def_kinds(self) -> &'static [(&'static str, crate::parse::SymKind)] {
        use crate::parse::SymKind::*;
        match self {
            Language::Rust => &[("function_item", Fn), ("function_signature_item", Fn), ("struct_item", Struct), ("enum_item", Enum), ("trait_item", Trait),
                ("impl_item", Impl), ("const_item", Const), ("static_item", Static), ("type_item", Type), ("mod_item", Module), ("macro_definition", Macro), ("union_item", Struct)],
            Language::Python => &[("function_definition", Fn), ("class_definition", Class)],
            Language::TypeScript | Language::Tsx | Language::JavaScript => &[("function_declaration", Fn), ("generator_function_declaration", Fn), ("class_declaration", Class),
                ("abstract_class_declaration", Class), ("method_definition", Method), ("interface_declaration", Interface), ("type_alias_declaration", Type),
                ("enum_declaration", Enum), ("lexical_declaration", Const), ("variable_declaration", Const), ("module", Module), ("internal_module", Module)],
            Language::Go => &[("function_declaration", Fn), ("method_declaration", Method), ("type_spec", Type), ("const_spec", Const), ("var_spec", Static)],
            Language::C => &[("function_definition", Fn), ("struct_specifier", Struct), ("enum_specifier", Enum), ("union_specifier", Struct), ("type_definition", Type), ("declaration", Static)],
            Language::Cpp => &[("function_definition", Fn), ("class_specifier", Class), ("struct_specifier", Struct), ("enum_specifier", Enum), ("namespace_definition", Module),
                ("type_definition", Type), ("alias_declaration", Type), ("template_declaration", Type)],
            Language::Java => &[("class_declaration", Class), ("interface_declaration", Interface), ("enum_declaration", Enum), ("method_declaration", Method),
                ("constructor_declaration", Method), ("record_declaration", Class)],
            Language::Json => &[("pair", Const)],
            Language::Toml => &[("table", Module), ("table_array_element", Module), ("pair", Const)],
            Language::Markdown => &[("atx_heading", Module), ("setext_heading", Module)],
        }
    }

    /// Leaf node kinds that count as references for the dependency graph.
    pub fn ref_kinds(self) -> &'static [&'static str] {
        match self {
            Language::Rust => &["identifier", "type_identifier", "field_identifier", "scoped_identifier"],
            Language::Python => &["identifier", "attribute"],
            Language::TypeScript | Language::Tsx | Language::JavaScript => &["identifier", "type_identifier", "property_identifier", "shorthand_property_identifier"],
            Language::Go => &["identifier", "type_identifier", "field_identifier", "package_identifier"],
            Language::C | Language::Cpp => &["identifier", "type_identifier", "field_identifier", "namespace_identifier"],
            Language::Java => &["identifier", "type_identifier"],
            Language::Json | Language::Toml | Language::Markdown => &[],
        }
    }

    /// Whether a file of this language participates in the reference graph (config/docs don't).
    pub fn is_code(self) -> bool { !matches!(self, Language::Json | Language::Toml | Language::Markdown) }
}
