use std::fmt::Debug;

use crate::address::{Address, Selection};

/// A trait for querying choice maps
pub trait ChoiceMapQuery<V> {
    /// Check if an address exists in this choice map
    fn contains(&self, addr: &Address) -> bool;

    /// Get the value at this address if it exists
    fn get_value(&self, addr: &Address) -> Option<&V>;

    /// Get the sub-choice-map at this address
    fn get_submap(&self, addr: &Address) -> Option<&Self>;

    /// Check if this node is a leaf (contains a value)
    fn is_leaf(&self) -> bool;

    /// Get the value if this is a leaf node
    fn as_value(&self) -> Option<&V>;

    /// Get direct child addresses (not all descendants)
    fn get_children_addresses(&self) -> Vec<Address>;

    /// Check if this choice map is empty
    fn is_empty(&self) -> bool;

    /// Get the total number of values in this choice map
    fn len(&self) -> usize;
}

/// A trait for mutable choice maps
pub trait ChoiceMap<V>: ChoiceMapQuery<V> + Clone + Debug + Default {
    type Iter<'a>: Iterator<Item = (Address, &'a V)>
    where
        Self: 'a,
        V: 'a;

    /// Set a value at the given address, creating intermediate nodes as needed
    fn set_value(&mut self, addr: Address, value: V);

    /// Remove the node at the given address
    fn remove(&mut self, addr: &Address) -> bool;

    /// Filter this choice map using a Selection
    fn filter(&self, selection: &Selection) -> Self;

    /// Create a selection that includes all addresses in this choice map
    fn to_selection(&self) -> Selection {
        Selection::from(self.get_all_addresses())
    }

    /// Get all addresses (not just direct children)
    fn get_all_addresses(&self) -> Vec<Address>;

    /// Iterate over all (address, value) pairs in this choice map
    fn iter(&self) -> Self::Iter<'_>;
}
