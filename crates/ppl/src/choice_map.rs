use std::fmt::Debug;

use crate::address::{Address, Selection};

/// Object-safe trait for iteration and querying
pub trait ChoiceMapQuery<V> {
    /// Get the node at this address - could be a value or internal structure
    fn get_dyn(&self, addr: &Address) -> Option<&dyn ChoiceMapQuery<V>>;

    /// Check if this node contains a value (is a leaf)
    fn is_leaf(&self) -> bool;

    /// Get the value if this is a leaf node
    fn as_value(&self) -> Option<&V>;

    /// Get all direct child keys if this is an internal node
    fn keys(&self) -> Vec<Address>;

    /// Check if this choice map is empty
    fn is_empty(&self) -> bool;
}

/// Full trait with non-object-safe methods
pub trait ChoiceMap<V>: ChoiceMapQuery<V> + Clone + Debug {
    type Iter<'a>: Iterator<Item = (Address, &'a V)>
    where
        Self: 'a,
        V: 'a;

    /// Get the node at this address - returns concrete type
    fn get(&self, addr: &Address) -> Option<&Self>;

    /// Set a value at the given address, creating intermediate nodes as needed
    fn set_value(&mut self, addr: Address, value: V);

    /// Remove the node at the given address
    fn remove(&mut self, addr: &Address) -> bool;

    /// Create an empty choice map
    fn empty() -> Self;

    /// Filter this choice map using a Selection
    fn filter(&self, selection: &Selection) -> Self;

    /// Create a selection that includes all addresses in this choice map
    fn selection(&self) -> Selection {
        Selection::from(self.keys())
    }

    /// Iterate over all (address, value) pairs in this choice map
    fn iter(&self) -> Self::Iter<'_>;
}
