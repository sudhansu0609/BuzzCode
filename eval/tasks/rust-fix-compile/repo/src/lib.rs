//! Inventory of items with quantities.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub name: String,
    pub qty: u32,
}

#[derive(Default)]
pub struct Inventory {
    items: HashMap<String, Item>,
}

impl Inventory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `qty` of `name`, creating the item if needed.
    pub fn add(&mut self, name: &str, qty: u32) {
        let entry = self.items.entry(name.to_string()).or_insert(Item { name: name.to_string(), qty: 0 });
        entry.qty += qty;
    }

    /// Remove up to `qty`; returns the amount actually removed.
    pub fn remove(&mut self, name: &str, qty: u32) -> u32 {
        match self.items.get_mut(name) {
            Some(item) => {
                let removed = qty.min(item.qty);
                item.qty -= removed;
                removed
            }
            None => 0,
        }
    }

    /// Names of items with a positive quantity, sorted.
    pub fn in_stock(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.items.values().filter(|i| i.qty > 0).map(|i| i.name).collect();
        names.sort();
        names
    }

    /// Total quantity across all items.
    pub fn total(&self) -> u32 {
        self.items.values().map(|i| i.qty).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_remove_total() {
        let mut inv = Inventory::new();
        inv.add("bolt", 5);
        inv.add("nut", 2);
        assert_eq!(inv.remove("bolt", 10), 5);
        assert_eq!(inv.total(), 2);
        assert_eq!(inv.in_stock(), vec!["nut"]);
    }
}
