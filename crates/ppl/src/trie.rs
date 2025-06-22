use crate::address::Address;
use std::collections::HashMap;

/// A Trie data structure that uses Address as keys
/// Separates leaf nodes (values) from internal nodes (sub-tries)
#[derive(Debug, Clone)]
pub struct Trie<T> {
    leaf: HashMap<Address, T>,
    internal: HashMap<Address, Trie<T>>,
}

/// Iterator over trie key-value pairs
pub struct TrieIter<'a, T> {
    stack: Vec<TrieIterFrame<'a, T>>,
}

struct TrieIterFrame<'a, T> {
    current_path: Vec<Address>,
    leaf_iter: std::collections::hash_map::Iter<'a, Address, T>,
    internal_iter: std::collections::hash_map::Iter<'a, Address, Trie<T>>,
    current_internal: Option<(&'a Address, &'a Trie<T>)>,
}

impl<'a, T> Iterator for TrieIter<'a, T> {
    type Item = (Vec<Address>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(frame) = self.stack.last_mut() {
            // First, try to get the next leaf node at the current level
            if let Some((addr, value)) = frame.leaf_iter.next() {
                let mut path = frame.current_path.clone();
                path.push(addr.clone());
                return Some((path, value));
            }

            // If no more leaf nodes, try to descend into internal nodes
            if let Some((addr, subtrie)) = frame.current_internal.take() {
                // Create a new frame for the subtrie
                let mut new_path = frame.current_path.clone();
                new_path.push(addr.clone());

                let new_frame = TrieIterFrame {
                    current_path: new_path,
                    leaf_iter: subtrie.leaf.iter(),
                    internal_iter: subtrie.internal.iter(),
                    current_internal: None,
                };
                self.stack.push(new_frame);
                continue;
            }

            // Try to get the next internal node
            if let Some((addr, subtrie)) = frame.internal_iter.next() {
                frame.current_internal = Some((addr, subtrie));
                continue;
            }

            // No more nodes at this level, pop the frame
            self.stack.pop();
        }

        None
    }
}

impl<T: Clone> Trie<T> {
    /// Create a new empty Trie
    pub fn new() -> Self {
        Self {
            leaf: HashMap::new(),
            internal: HashMap::new(),
        }
    }

    /// Check if the trie is empty (invariant: all internal nodes are nonempty)
    pub fn is_empty(&self) -> bool {
        self.leaf.is_empty() && self.internal.is_empty()
    }

    /// Get the leaf nodes HashMap
    pub fn get_leaf_nodes(&self) -> &HashMap<Address, T> {
        &self.leaf
    }

    /// Get the internal nodes HashMap
    pub fn get_internal_nodes(&self) -> &HashMap<Address, Trie<T>> {
        &self.internal
    }

    /// Check if a leaf node exists at the given address
    pub fn has_leaf_node(&self, addr: &Address) -> bool {
        self.leaf.contains_key(addr)
    }

    /// Check if a leaf node exists at the given address path
    pub fn has_leaf_node_path(&self, path: &[Address]) -> bool {
        if path.is_empty() {
            false
        } else if path.len() == 1 {
            self.has_leaf_node(&path[0])
        } else {
            let first = &path[0];
            let rest = &path[1..];
            self.internal
                .get(first)
                .map(|node| node.has_leaf_node_path(rest))
                .unwrap_or(false)
        }
    }

    /// Get a leaf node value at the given address
    pub fn get_leaf_node(&self, addr: &Address) -> Option<&T> {
        self.leaf.get(addr)
    }

    /// Get a leaf node value at the given address path
    pub fn get_leaf_node_path(&self, path: &[Address]) -> Option<&T> {
        if path.is_empty() {
            None
        } else if path.len() == 1 {
            self.get_leaf_node(&path[0])
        } else {
            let first = &path[0];
            let rest = &path[1..];
            self.internal
                .get(first)
                .and_then(|node| node.get_leaf_node_path(rest))
        }
    }

    /// Set a leaf node value at the given address
    pub fn set_leaf_node(&mut self, addr: Address, value: T) {
        self.leaf.insert(addr, value);
    }

    /// Set a leaf node value at the given address path
    pub fn set_leaf_node_path(&mut self, path: &[Address], value: T) {
        if path.is_empty() {
            return;
        } else if path.len() == 1 {
            self.set_leaf_node(path[0].clone(), value);
        } else {
            let first = &path[0];
            let rest = &path[1..];
            let node = self.internal.entry(first.clone()).or_insert_with(Trie::new);
            node.set_leaf_node_path(rest, value);
        }
    }

    /// Delete a leaf node at the given address
    pub fn delete_leaf_node(&mut self, addr: &Address) -> Option<T> {
        let result = self.leaf.remove(addr);
        result
    }

    /// Delete a leaf node at the given address path
    pub fn delete_leaf_node_path(&mut self, path: &[Address]) -> Option<T> {
        if path.is_empty() {
            None
        } else if path.len() == 1 {
            self.delete_leaf_node(&path[0])
        } else {
            let first = &path[0];
            let rest = &path[1..];
            if let Some(node) = self.internal.get_mut(first) {
                let result = node.delete_leaf_node_path(rest);
                // Clean up empty internal nodes
                if node.is_empty() {
                    self.internal.remove(first);
                }
                result
            } else {
                None
            }
        }
    }

    /// Check if an internal node exists at the given address
    pub fn has_internal_node(&self, addr: &Address) -> bool {
        self.internal.contains_key(addr)
    }

    /// Check if an internal node exists at the given address path
    pub fn has_internal_node_path(&self, path: &[Address]) -> bool {
        if path.is_empty() {
            false
        } else if path.len() == 1 {
            self.has_internal_node(&path[0])
        } else {
            let first = &path[0];
            let rest = &path[1..];
            self.internal
                .get(first)
                .map(|node| node.has_internal_node_path(rest))
                .unwrap_or(false)
        }
    }

    /// Get an internal node at the given address
    pub fn get_internal_node(&self, addr: &Address) -> Option<&Trie<T>> {
        self.internal.get(addr)
    }

    /// Get an internal node at the given address path
    pub fn get_internal_node_path(&self, path: &[Address]) -> Option<&Trie<T>> {
        if path.is_empty() {
            None
        } else if path.len() == 1 {
            self.get_internal_node(&path[0])
        } else {
            let first = &path[0];
            let rest = &path[1..];
            self.internal
                .get(first)
                .and_then(|node| node.get_internal_node_path(rest))
        }
    }

    /// Set an internal node at the given address
    pub fn set_internal_node(&mut self, addr: Address, new_node: Trie<T>) {
        if !new_node.is_empty() {
            self.internal.insert(addr, new_node);
        }
    }

    /// Set an internal node at the given address path
    pub fn set_internal_node_path(&mut self, path: &[Address], new_node: Trie<T>) {
        if path.is_empty() {
            return;
        } else if path.len() == 1 {
            self.set_internal_node(path[0].clone(), new_node);
        } else {
            let first = &path[0];
            let rest = &path[1..];
            let node = self.internal.entry(first.clone()).or_insert_with(Trie::new);
            node.set_internal_node_path(rest, new_node);
        }
    }

    /// Delete an internal node at the given address
    pub fn delete_internal_node(&mut self, addr: &Address) -> Option<Trie<T>> {
        self.internal.remove(addr)
    }

    /// Delete an internal node at the given address path
    pub fn delete_internal_node_path(&mut self, path: &[Address]) -> Option<Trie<T>> {
        if path.is_empty() {
            None
        } else if path.len() == 1 {
            self.delete_internal_node(&path[0])
        } else {
            let first = &path[0];
            let rest = &path[1..];
            if let Some(node) = self.internal.get_mut(first) {
                let result = node.delete_internal_node_path(rest);
                // Clean up empty internal nodes
                if node.is_empty() {
                    self.internal.remove(first);
                }
                result
            } else {
                None
            }
        }
    }

    /// Insert a value at the given address path (convenience method)
    pub fn insert(&mut self, path: &[Address], value: T) {
        self.set_leaf_node_path(path, value);
    }

    /// Get a value at the given address path (convenience method)
    pub fn get(&self, path: &[Address]) -> Option<&T> {
        self.get_leaf_node_path(path)
    }

    /// Check if a value exists at the given address path (convenience method)
    pub fn contains(&self, path: &[Address]) -> bool {
        self.has_leaf_node_path(path)
    }

    /// Remove a value at the given address path (convenience method)
    pub fn remove(&mut self, path: &[Address]) -> Option<T> {
        self.delete_leaf_node_path(path)
    }

    /// Get an iterator over all key-value pairs in the trie
    pub fn iter(&self) -> TrieIter<T> {
        TrieIter {
            stack: vec![TrieIterFrame {
                current_path: Vec::new(),
                leaf_iter: self.leaf.iter(),
                internal_iter: self.internal.iter(),
                current_internal: None,
            }],
        }
    }

    /// Get all key-value pairs in the trie (legacy method for backward compatibility)
    pub fn iter_vec(&self) -> Vec<(Vec<Address>, &T)> {
        self.iter().collect()
    }

    /// Get all values in the trie
    pub fn values(&self) -> Vec<&T> {
        let mut result = Vec::new();
        self.collect_values(&mut result);
        result
    }

    /// Helper method to recursively collect values
    fn collect_values<'a>(&'a self, result: &mut Vec<&'a T>) {
        // Collect values from leaf nodes
        for value in self.leaf.values() {
            result.push(value);
        }

        // Recurse into internal nodes
        for child in self.internal.values() {
            child.collect_values(result);
        }
    }

    /// Get all keys in the trie
    pub fn keys(&self) -> Vec<Vec<Address>> {
        let mut result = Vec::new();
        self.collect_keys(&mut result, Vec::new());
        result
    }

    /// Helper method to recursively collect keys
    fn collect_keys(&self, result: &mut Vec<Vec<Address>>, current_path: Vec<Address>) {
        // Collect keys from leaf nodes
        for addr in self.leaf.keys() {
            let mut path = current_path.clone();
            path.push(addr.clone());
            result.push(path);
        }

        // Recurse into internal nodes
        for (addr, child) in &self.internal {
            let mut new_path = current_path.clone();
            new_path.push(addr.clone());
            child.collect_keys(result, new_path);
        }
    }

    /// Merge another trie into this one
    pub fn merge(&mut self, other: Trie<T>) {
        // Merge leaf nodes
        for (key, value) in other.leaf {
            self.leaf.insert(key, value);
        }

        // Merge internal nodes
        for (key, other_subtrie) in other.internal {
            if let Some(existing_subtrie) = self.internal.get_mut(&key) {
                existing_subtrie.merge(other_subtrie);
            } else {
                self.internal.insert(key, other_subtrie);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{path, sym};

    #[test]
    fn test_trie_basic_operations() {
        let mut trie = Trie::new();

        // Test empty trie
        assert!(trie.is_empty());
        assert_eq!(trie.get(&[sym!(x)]), None);

        // Test inserting and getting values
        trie.insert(&[sym!(x)], 1);
        assert!(!trie.is_empty());
        assert_eq!(trie.get(&[sym!(x)]), Some(&1));

        // Test inserting at nested paths
        trie.insert(&[sym!(x), sym!(y)], 2);
        assert_eq!(trie.get(&[sym!(x), sym!(y)]), Some(&2));

        // Test single address operations
        trie.set_leaf_node(sym!(z), 3);
        assert_eq!(trie.get_leaf_node(&sym!(z)), Some(&3));
    }

    #[test]
    fn test_trie_remove() {
        let mut trie = Trie::new();

        trie.insert(&[sym!(x)], 1);
        trie.insert(&[sym!(x), sym!(y)], 2);

        // Test removing leaf node
        assert_eq!(trie.remove(&[sym!(x), sym!(y)]), Some(2));
        assert!(!trie.contains(&[sym!(x), sym!(y)]));
        assert!(trie.contains(&[sym!(x)]));

        // Test removing non-existent path
        assert_eq!(trie.remove(&[sym!(z)]), None);
    }

    #[test]
    fn test_trie_iter() {
        let mut trie = Trie::new();

        trie.insert(&[sym!(x)], 1);
        trie.insert(&[sym!(x), sym!(y)], 2);
        trie.insert(&[sym!(z)], 3);

        let pairs: Vec<_> = trie.iter().collect();
        assert_eq!(pairs.len(), 3);

        // Test values
        let values = trie.values();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&&1));
        assert!(values.contains(&&2));
        assert!(values.contains(&&3));

        // Test keys
        let keys = trie.keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&vec![sym!(x)]));
        assert!(keys.contains(&vec![sym!(x), sym!(y)]));
        assert!(keys.contains(&vec![sym!(z)]));
    }

    #[test]
    fn test_trie_internal_nodes() {
        let mut trie = Trie::new();

        // Create a subtrie
        let mut subtrie = Trie::new();
        subtrie.set_leaf_node(sym!(a), 42);

        // Set it as an internal node
        trie.set_internal_node(sym!(root), subtrie);

        // Check internal node operations
        assert!(trie.has_internal_node(&sym!(root)));
        let retrieved = trie.get_internal_node(&sym!(root)).unwrap();
        assert_eq!(retrieved.get_leaf_node(&sym!(a)), Some(&42));
    }

    #[test]
    fn test_trie_merge() {
        let mut trie1 = Trie::new();
        trie1.insert(&[sym!(x)], 1);
        trie1.insert(&[sym!(y)], 2);

        let mut trie2 = Trie::new();
        trie2.insert(&[sym!(z)], 3);
        trie2.insert(&[sym!(x), sym!(a)], 4);

        trie1.merge(trie2);

        assert_eq!(trie1.get(&[sym!(x)]), Some(&1));
        assert_eq!(trie1.get(&[sym!(y)]), Some(&2));
        assert_eq!(trie1.get(&[sym!(z)]), Some(&3));
        assert_eq!(trie1.get(&[sym!(x), sym!(a)]), Some(&4));
    }

    #[test]
    fn test_trie_with_path_addresses() {
        let mut trie = Trie::new();

        let path1 = path!(x, y);
        let path2 = path!(a, b, c);

        trie.insert(&[path1.clone()], 1);
        trie.insert(&[path2.clone()], 2);

        assert_eq!(trie.get(&[path1]), Some(&1));
        assert_eq!(trie.get(&[path2]), Some(&2));
    }
}
