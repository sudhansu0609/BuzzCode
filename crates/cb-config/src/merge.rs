use toml::Value;

/// Deep-merge `layer` into `base`. Tables merge recursively; arrays of tables with a `name`
/// key merge by name (so a project config can override one `[[profile]]` field); everything
/// else is replaced.
pub fn merge_into(base: &mut Value, layer: Value) {
    match (base, layer) {
        (Value::Table(b), Value::Table(l)) => {
            for (k, v) in l {
                match b.get_mut(&k) {
                    Some(existing) => merge_into(existing, v),
                    None => { b.insert(k, v); }
                }
            }
        }
        (Value::Array(b), Value::Array(l)) if is_named_table_array(b) && is_named_table_array(&l) => {
            for item in l {
                let name = item.get("name").and_then(Value::as_str).map(str::to_owned);
                match name.and_then(|n| b.iter_mut().find(|e| e.get("name").and_then(Value::as_str) == Some(&n))) {
                    Some(existing) => merge_into(existing, item),
                    None => b.push(item),
                }
            }
        }
        (b, l) => *b = l,
    }
}

fn is_named_table_array(arr: &[Value]) -> bool {
    !arr.is_empty() && arr.iter().all(|v| v.is_table() && v.get("name").is_some())
}

/// A handful of environment overrides that are convenient for scripting / CI.
pub fn apply_env(doc: &mut Value) {
    let set = |doc: &mut Value, path: &[&str], v: Value| {
        let mut cur = doc;
        for (i, key) in path.iter().enumerate() {
            if i + 1 == path.len() {
                if let Value::Table(t) = cur { t.insert((*key).to_string(), v); }
                return;
            }
            if !cur.is_table() { return; }
            let t = cur.as_table_mut().unwrap();
            cur = t.entry((*key).to_string()).or_insert_with(|| Value::Table(Default::default()));
        }
    };
    if let Ok(v) = std::env::var("BUZZCODE_PROFILE") { set(doc, &["general", "default_profile"], Value::String(v)); }
    if let Ok(v) = std::env::var("BUZZCODE_PERMISSION_MODE") { set(doc, &["general", "permission_mode"], Value::String(v)); }
    if let Ok(v) = std::env::var("BUZZCODE_LOG_LEVEL") { set(doc, &["general", "log_level"], Value::String(v)); }
    if let Ok(v) = std::env::var("BUZZCODE_ENGINE_URL") {
        set(doc, &["engine", "provider"], Value::String("external".into()));
        set(doc, &["engine", "external_url"], Value::String(v));
    }
    if let Ok(v) = std::env::var("BUZZCODE_ENGINE_PORT") {
        if let Ok(p) = v.parse::<i64>() { set(doc, &["engine", "port"], Value::Integer(p)); }
    }
    if let Ok(v) = std::env::var("BUZZCODE_LLAMA_SERVER") { set(doc, &["engine", "server_binary"], Value::String(v)); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_named_arrays_by_name() {
        let mut base: Value = toml::from_str(r#"
            [[profile]]
            name = "a"
            ctx = 1
            [[profile]]
            name = "b"
            ctx = 2
        "#).unwrap();
        let layer: Value = toml::from_str(r#"
            [[profile]]
            name = "b"
            ctx = 99
            [[profile]]
            name = "c"
            ctx = 3
        "#).unwrap();
        merge_into(&mut base, layer);
        let arr = base["profile"].as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[1]["ctx"].as_integer(), Some(99));
        assert_eq!(arr[2]["name"].as_str(), Some("c"));
    }

    #[test]
    fn deep_merges_tables() {
        let mut base: Value = toml::from_str("[engine]\nport = 1\nhost = 'x'").unwrap();
        let layer: Value = toml::from_str("[engine]\nport = 2").unwrap();
        merge_into(&mut base, layer);
        assert_eq!(base["engine"]["port"].as_integer(), Some(2));
        assert_eq!(base["engine"]["host"].as_str(), Some("x"));
    }
}
