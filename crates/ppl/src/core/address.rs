use std::fmt::{Debug, Display};
use std::collections::HashMap;
use std::collections::HashSet;

// Address can be single component or multiple components (tuple)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Address {
    Symbol(String),
    Index(i32),
    Wildcard,
    Path(Vec<Address>),
}


// From implementations for easy construction
impl From<&str> for Address {
    fn from(s: &str) -> Self {
        Address::Symbol(s.to_string())
    }
}

impl From<String> for Address {
    fn from(s: String) -> Self {
        Address::Symbol(s)
    }
}

impl From<i32> for Address {
    fn from(i: i32) -> Self {
        Address::Index(i)
    }
}

// Display implementations for debugging
impl Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Address::Symbol(s) => write!(f, "{}", s),
            Address::Index(i) => write!(f, "{}", i),
            Address::Path(path) => {
                write!(f, "[{}]", path.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", "))
            }
            Address::Wildcard => write!(f, "*"),
        }
    }
}

impl Address {
    pub fn is_static(&self) -> bool {
        match self {
            Address::Symbol(_) => true,
            Address::Index(_) => false,
            Address::Wildcard => true,
            Address::Path(components) => components.iter().all(|c| c.is_static())
        }
    }
}

/// Selection types - simplified and more functional
#[derive(Clone, PartialEq, Debug)]
pub enum Selection {
    All,
    None,
    Leaf,
    Static(Box<Selection>, Address),
    Complement(Box<Selection>),
    And(Box<Selection>, Box<Selection>),
    Or(Box<Selection>, Box<Selection>)
}

impl Selection {

    pub fn check(&self) -> bool {
        match self {
            Selection::All => true,
            Selection::None => false, 
            Selection::Leaf => true,
            Selection::Static(_,_) => false,  // ← This should be false!
            Selection::Complement(inner) => !inner.check(),
            Selection::And(left, right) => left.check() && right.check(),
            Selection::Or(left, right) => left.check() || right.check(),
        }
    }

    /// Navigate to a subselection at the given address
    /// Equivalent to Python's __call__ method
    pub fn call(&self, addr: Address) -> Selection {
        match addr {
            // For path addresses, navigate through each component
            Address::Path(components) => {
                let mut subselection = self.clone();
                for comp in components {
                    subselection = subselection.get_subselection(&comp);
                }
                subselection
            }
            // For single component addresses, navigate directly
            single_component => {
                self.get_subselection(&single_component)
            }
        }
    }

    /// Check if an address is contained in this selection
    pub fn contains(&self, addr: &Address) -> bool {
        self.call(addr.clone()).check()
    }
    
    /// Extend this selection by prefixing it with the given address components
    /// This creates nested Static selections in reverse order
    pub fn extend(&self, addrs: &[Address]) -> Selection {
        let mut acc = self.clone();
        for addr in addrs.iter().rev() {
            
            acc = match acc {
                Selection::None => acc,
                _ => Selection::Static(Box::new(acc), addr.clone()),
            }
        }
        acc
    }
    
    /// Get the subselection at the given address
    pub fn get_subselection(&self, addr: &Address) -> Selection {
        match self {
            Selection::All => self.clone(),
            Selection::None => self.clone(),
            Selection::Leaf => Selection::None,
            Selection::Complement(inner) => {
                !inner.call(addr.clone())
            }
            Selection::Static(inner, target) => {
                match target {
                    Address::Wildcard => *inner.clone(),
                    _ if target == addr => *inner.clone(),
                    _ => Selection::None,
                }
            }
            Selection::And(left, right) => {
                left.call(addr.clone()) & right.call(addr.clone())
            }
            Selection::Or(left, right) => {
                left.call(addr.clone()) | right.call(addr.clone())
            }
        }
    }
}


impl std::ops::Index<Address> for Selection {
    type Output = bool;
    
    /// Index operation returns whether the address is contained in the selection
    fn index(&self, addr: Address) -> &Self::Output {
        if self.call(addr).check() {
            &true
        } else {
            &false
        }
    }
}


impl std::ops::BitOr for Selection {
    type Output = Selection;
    
    fn bitor(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Selection::All, _) => self,
            (_, Selection::All) => rhs,
            (Selection::None, _) => rhs,
            (_, Selection::None) => self,
            _ if self == rhs => self,
            _ => Selection::Or(Box::new(self), Box::new(rhs)),
        }
    }
}


impl std::ops::BitAnd for Selection {
    type Output = Selection;
    
    fn bitand(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Selection::All, _) => rhs,
            (_, Selection::All) => self,
            (Selection::None, _) => self,
            (_, Selection::None) => rhs,
            _ if self == rhs => self,
            _ => Selection::And(Box::new(self), Box::new(rhs)),
        }
    }
}


impl std::ops::Not for Selection {
    type Output = Selection;
    
    fn not(self) -> Self::Output {
        match self {
            // Basic complements
            Selection::All => Selection::None,
            Selection::None => Selection::All,
            // Double negation
            Selection::Complement(inner) => *inner,
            
            // Other cases
            other => Selection::Complement(Box::new(other)),
        }
    }
}

/// Macro to create selections
#[macro_export]
macro_rules! s {
    ($x:ident) => {
        Selection::Static(Box::new(Selection::All), Address::Symbol(stringify!($x).to_string()))
    };
    (*) => {
        Selection::Static(Address::Wildcard)
    };
    ($($x:ident),+ $(,)?) => {{
        let mut result = Selection::Leaf;
        let components = vec![$(Address::Symbol(stringify!($x).to_string())),+];
        for addr in components.into_iter().rev() {
            result = Selection::Static(Box::new(result), addr);
        }
        result
    }};
    (*, $($x:ident),+ $(,)?) => {{
        let mut result = Selection::All;
        let mut components = vec![$(Address::Symbol(stringify!($x).to_string())),+];
        components.insert(0, Address::Wildcard);
        for addr in components.into_iter().rev() {
            result = Selection::Static(Box::new(result), addr);
        }
        result
    }};
}

/// Macro to create symbol addresses
#[macro_export]
macro_rules! sym {
    ($x:ident) => {
        Address::Symbol(stringify!($x).to_string())
    };
    ($x:expr) => {
        Address::Symbol($x.to_string())
    };
}

/// Macro to create path addresses
#[macro_export]
macro_rules! path {
    () => {
        Address::Path(vec![])
    };
    ($($x:ident),+ $(,)?) => {
        Address::Path(vec![$(sym!($x)),+])
    };
    ($($x:expr),+ $(,)?) => {
        Address::Path(vec![$($x),+])
    };
}



/// Choice map implementation


/// A type that combines a value with a boolean flag indicating whether the value is active
#[derive(Debug, Clone)]
pub struct Mask<T: Clone> {
    value: T,
    flag: bool,
}

impl<T: Clone> Mask<T> {
    /// Create a new Mask with a value and flag
    pub fn new(value: T, flag: bool) -> Self {
        Self { value, flag }
    }

    /// Get the value if the flag is true, otherwise None
    pub fn get_value(&self) -> Option<&T> {
        if self.flag {
            Some(&self.value)
        } else {
            None
        }
    }

    /// Get the raw value regardless of flag
    pub fn get_raw_value(&self) -> &T {
        &self.value
    }

    /// Get the flag value
    pub fn get_flag(&self) -> bool {
        self.flag
    }

    /// Combine multiple masks using OR operation
    /// Returns the first value that has a true flag, or None if all flags are false
    pub fn or_n(masks: &[Self]) -> Option<&T> {
        masks.iter()
            .find(|mask| mask.flag)
            .map(|mask| &mask.value)
    }
}

// Implement OR operator for Mask
impl<T: Clone> std::ops::BitOr for Mask<T> {
    type Output = Self;
    
    fn bitor(self, rhs: Self) -> Self::Output {
        if self.flag {
            self
        } else {
            rhs
        }
    }
}



/// Not supporting dynamic indexing yet
#[derive(Debug, Clone, PartialEq)]
pub enum ChoiceMap<T: Clone> {
    Empty,
    Choice(T),
    Static(HashMap<Address, ChoiceMap<T>>),
    Switch(i32, Vec<ChoiceMap<T>>),
    Or(Box<ChoiceMap<T>>, Box<ChoiceMap<T>>),
}




impl<T: Clone + Debug> ChoiceMap<T> {

    /// Get the value at an address
    pub fn get(&self, addr: &Address) -> T {
        self.get_submap(addr).get_value().unwrap_or_else(|| 
            panic!("No value found at address: {:?}", addr)
        )
    }

    /// Iterate over all address-value pairs in this ChoiceMap
    pub fn iter(&self) -> Vec<(Address, T)> {
        let mut result = Vec::new();
        self.collect_pairs(&mut result, Vec::new());
        result
    }

    /// Helper method to recursively collect address-value pairs
    fn collect_pairs(&self, result: &mut Vec<(Address, T)>, path: Vec<Address>) {
        match self {
            ChoiceMap::Empty => {},
            ChoiceMap::Choice(value) => {
                let addr = if path.is_empty() {
                    Address::Path(vec![])
                } else if path.len() == 1 {
                    path[0].clone()
                } else {
                    Address::Path(path)
                };
                result.push((addr, value.clone()));
            },
            ChoiceMap::Static(map) => {
                for (addr, submap) in map {
                    let mut new_path = path.clone();
                    new_path.push(addr.clone());
                    submap.collect_pairs(result, new_path);
                }
            },
            ChoiceMap::Switch(idx, choices) => {
                if *idx >= 0 && (*idx as usize) < choices.len() {
                    choices[*idx as usize].collect_pairs(result, path);
                }
            },
            ChoiceMap::Or(left, right) => {
                left.collect_pairs(result, path.clone());
                right.collect_pairs(result, path);
            },
        }
    }

    /// Check if this choice map has a value
    pub fn has_value(&self) -> bool {
        self.get_value().is_some()
    }
    
    /// Get value at the root (if this is a Choice)
    pub fn get_value(&self) -> Option<T> {
        match &self {
            ChoiceMap::Static(_) => None,
            ChoiceMap::Switch(_, choices) => choices.iter().find_map(|choice| choice.get_value()),
            ChoiceMap::Choice(value) => Some(value.clone()),
            _ => None,
        }
    }

    /// Get the inner map for a single address component
    pub fn get_inner_map(&self, addr: &Address) -> ChoiceMap<T> {
        match (self, addr) {
            (ChoiceMap::Empty, _) => ChoiceMap::Empty,
            (ChoiceMap::Choice(_), addr) => {
                if addr.is_static() {
                    ChoiceMap::Empty
                } else {
                    // TODO: Implement tree_map for dynamic addresses
                    ChoiceMap::Empty
                }
            }
            (ChoiceMap::Static(map), addr) => {
                if addr.is_static() {
                    map.get(addr).cloned().unwrap_or(ChoiceMap::Empty)
                } else {
                    // TODO: Implement tree_map for dynamic addresses
                    ChoiceMap::Empty
                }
            }
            (ChoiceMap::Switch(idx, choices), _) => {
                if *idx >= 0 && (*idx as usize) < choices.len() {
                    choices[*idx as usize].clone()
                } else {
                    ChoiceMap::Empty
                }
            }
            (ChoiceMap::Or(left, right), _) => {
                left.get_inner_map(addr) | right.get_inner_map(addr)
            }
        }
    }

    /// Get submap by following a path of addresses
    pub fn get_submap(&self, addr: &Address) -> ChoiceMap<T> {
        match addr {
            Address::Path(components) => {
                let mut current = self.clone();
                for component in components {
                    current = current.get_inner_map(component);
                }
                current
            }
            single_addr => self.get_inner_map(single_addr)
        }
    }

    /// Filter this choice map using a selection
    pub fn filter(&self, selection: &Selection) -> ChoiceMap<T> {
        match self {
            ChoiceMap::Empty => ChoiceMap::Empty,
            ChoiceMap::Choice(value) => {
                if selection.check() {
                    ChoiceMap::Choice(value.clone())
                } else {
                    ChoiceMap::Empty
                }
            }
            ChoiceMap::Static(map) => {
                let mut filtered_map = HashMap::new();
                for (addr, submap) in map {
                    let sub_selection = selection.call(addr.clone());
                    let filtered_submap = submap.filter(&sub_selection);
                    if !filtered_submap.is_empty() {
                        filtered_map.insert(addr.clone(), filtered_submap);
                    }
                }
                if filtered_map.is_empty() {
                    ChoiceMap::Empty
                } else {
                    ChoiceMap::Static(filtered_map)
                }
            },
            ChoiceMap::Switch(idx, choices) => {
                let chms: Vec<_> = choices.iter()
                    .map(|choice| choice.filter(selection))
                    .collect();
                if *idx >= 0 && (*idx as usize) < chms.len() {
                    chms[*idx as usize].clone()
                } else {
                    ChoiceMap::Switch(*idx, chms)
                }
            }
            ChoiceMap::Or(left, right) => {
                left.filter(selection) | right.filter(selection)
            }
        }
    }
    
    /// Merge two choice maps (OR operation with first taking priority)
    pub fn merge(&self, other: &ChoiceMap<T>) -> ChoiceMap<T> {
        if self.is_empty() {
            other.clone()
        } else if other.is_empty() {
            self.clone()
        } else {
            match (self, other) {
                (ChoiceMap::Static(map1), ChoiceMap::Static(map2)) => {
                    let mut merged_map = HashMap::new();
                    // Get all unique keys from both maps
                    let keys: HashSet<_> = map1.keys().chain(map2.keys()).cloned().collect();
                    
                    for key in keys {
                        match (map1.get(&key), map2.get(&key)) {
                            (Some(submap1), Some(submap2)) => {
                                merged_map.insert(key, submap1.merge(submap2));
                            }
                            (Some(submap1), None) => {
                                merged_map.insert(key, submap1.clone());
                            }
                            (None, Some(submap2)) => {
                                merged_map.insert(key, submap2.clone());
                            }
                            (None, None) => unreachable!(),
                        }
                    }
                    ChoiceMap::Static(merged_map)
                }
                (ChoiceMap::Choice(a), ChoiceMap::Choice(b)) => {
                    let a = Mask::new(a, true);
                    let b = Mask::new(b, true);
                    ChoiceMap::Choice((a | b).value.clone())
                }
                (ChoiceMap::Switch(idx, chms), _) => {
                    let new_chms: Vec<_> = chms.iter()
                        .map(|c1| c1.clone() | other.clone())
                        .collect();
                    ChoiceMap::Switch(*idx, new_chms)
                }
                (_, ChoiceMap::Switch(idx, chms)) => {
                    let new_chms: Vec<_> = chms.iter()
                        .map(|c2| self.clone() | c2.clone())
                        .collect();
                    ChoiceMap::Switch(*idx, new_chms)
                }
                (ChoiceMap::Choice(_), _) | (_, ChoiceMap::Choice(_)) => {
                    panic!("Cannot handle Choice and non-Choice in Or")
                }
                _ => ChoiceMap::Or(Box::new(self.clone()), Box::new(other.clone())),
            }
        }
    }
    
    /// Check if an address is contained in this choice map
    pub fn contains(&self, addr: &Address) -> bool {
        self.get_submap(addr).has_value()
    }
    
    /// Create a choice map with a value at a specific address path
    pub fn entry(value: T, addrs: &[Address]) -> Self {
        let mut result = ChoiceMap::Choice(value);
        for addr in addrs.iter().rev() {
            let mut map = HashMap::new();
            map.insert(addr.clone(), result);
            result = ChoiceMap::Static(map);
        }
        result
    }
    
    /// Extend this choice map by nesting it under the given addresses
    pub fn extend(&self, addrs: &[Address]) -> Self {
        let mut result = self.clone();
        for addr in addrs.iter().rev() {
            let mut map = HashMap::new();
            map.insert(addr.clone(), result);
            result = ChoiceMap::Static(map);
        }
        result
    }
    
    /// Check if this choice map is empty
    pub fn is_empty(&self) -> bool {
        match self {
            ChoiceMap::Empty => true,
            ChoiceMap::Choice(_) => false,
            ChoiceMap::Static(map) => map.is_empty(),
            ChoiceMap::Switch(_, choices) => choices.iter().any(|choice| choice.is_empty()),
            ChoiceMap::Or(left, right) => left.is_empty() && right.is_empty(),
        }
    }

    /// Mask this choice map with a boolean flag
    /// If flag is true, returns the original ChoiceMap
    /// If flag is false, returns an empty ChoiceMap
    pub fn mask(&self, flag: bool) -> ChoiceMap<T> {
        if flag {
            self.clone()
        } else {
            ChoiceMap::Empty
        }
    }


    // Make a choice map that switches between multiple choice maps based on an index
    pub fn switch(idx: i32, chms: Vec<ChoiceMap<T>>) -> ChoiceMap<T> {
        chms[idx as usize].clone()
    }
}




// Implement OR operator for merging
impl<T: Clone + Debug> std::ops::BitOr for ChoiceMap<T> {
    type Output = Self;
    
    fn bitor(self, rhs: Self) -> Self::Output {
        self.merge(&rhs)
    }
}

/// Trait for creating choices
pub trait IntoChoiceMap<T: Clone> {
    fn choice_map(self) -> ChoiceMap<T>;
}

impl<T: Clone> IntoChoiceMap<T> for T {
    fn choice_map(self) -> ChoiceMap<T> {
        ChoiceMap::Choice(self)
    }
}

impl<T: Clone> IntoChoiceMap<T> for Mask<T> {
    fn choice_map(self) -> ChoiceMap<T> {
        match self.get_flag() {
            false => ChoiceMap::Empty,
            true => ChoiceMap::Choice(self.get_raw_value().clone()),
        }
    }
}

impl<T: Clone> IntoChoiceMap<T> for HashMap<Address, T> {
    fn choice_map(self) -> ChoiceMap<T> {
        let mut static_map = HashMap::new();
        for (addr, value) in self {
            static_map.insert(addr, ChoiceMap::Choice(value));
        }
        ChoiceMap::Static(static_map)
    }
}

impl<T: Clone> IntoChoiceMap<T> for HashMap<String, T> {
    fn choice_map(self) -> ChoiceMap<T> {
        let mut static_map = HashMap::new();
        for (addr, value) in self {
            static_map.insert(Address::Symbol(addr), ChoiceMap::Choice(value));
        }
        ChoiceMap::Static(static_map)
    }
}

impl<T: Clone + Debug> IntoChoiceMap<T> for Vec<(Address, T)> {
    fn choice_map(self) -> ChoiceMap<T> {
        let mut result = ChoiceMap::Empty;
        for (addr, value) in self {
            let entry_map = match addr {
                Address::Path(components) => {
                    ChoiceMap::entry(value, &components)
                }
                single_addr => {
                    ChoiceMap::entry(value, &[single_addr])
                }
            };
            result = result | entry_map;
        }
        result
    }
}


/// Macro to create choice maps
#[macro_export]
macro_rules! chm {
    () => {
        ChoiceMap::empty()
    };
    ($value:expr) => {
        ChoiceMap::choice($value)
    };
    ($value:expr; $($addr:expr),+ $(,)?) => {
        ChoiceMap::entry($value, &[$($addr),+])
    };
    // Dictionary syntax with mixed values and nested blocks
    ($($key:ident: $value:tt),+ $(,)?) => {{
        let mut map = HashMap::new();
        $(
            chm!(@insert map, $key, $value);
        )+
        ChoiceMap::Static(map)
    }};
    // Helper rule for inserting nested blocks
    (@insert $map:ident, $key:ident, { $($nested:tt)* }) => {
        $map.insert(sym!($key), chm!($($nested)*));
    };
    // Helper rule for inserting simple values
    (@insert $map:ident, $key:ident, $value:expr) => {
        $map.insert(sym!($key), ChoiceMap::choice($value));
    };
}

/// Builder for setting values in ChoiceMaps at specific paths
#[derive(Debug, Clone)]
pub struct ChoiceMapBuilder<T: Clone + Debug> {
    base: Option<ChoiceMap<T>>,
    path: Vec<Address>,
}

impl<T: Clone + Debug> ChoiceMapBuilder<T> {
    /// Create a new builder with a base ChoiceMap and path
    pub fn new(base: ChoiceMap<T>, path: Vec<Address>) -> Self {
        let base = if base.is_empty() { None } else { Some(base) };
        Self { base, path }
    }
    
    /// Create a new builder with no base ChoiceMap
    pub fn new_empty(path: Vec<Address>) -> Self {
        Self { base: None, path }
    }
    
    /// Chain another address component (equivalent to Python's __getitem__)
    pub fn at(&self, addr: Address) -> ChoiceMapBuilder<T> {
        let mut new_path = self.path.clone();
        new_path.push(addr);
        ChoiceMapBuilder {
            base: self.base.clone(),
            path: new_path,
        }
    }
    
    /// Chain multiple address components
    pub fn at_path(&self, addrs: &[Address]) -> ChoiceMapBuilder<T> {
        let mut builder = self.clone();
        for addr in addrs {
            builder = builder.at(addr.clone());
        }
        builder
    }
    
    /// Set a value at the current path
    pub fn set(self, value: T) -> ChoiceMap<T> {
        let entry_map = ChoiceMap::entry(value, &self.path);
        match self.base {
            None => entry_map,
            Some(base) => entry_map | base,
        }
    }
    
    /// Update an existing value at the current address using a function
    /// The function receives the current value (or ChoiceMap::Empty if none exists)
    pub fn update<F>(self, f: F) -> ChoiceMap<T>
    where
        F: FnOnce(ChoiceMap<T>) -> T,
    {
        let current_value = match &self.base {
            None => ChoiceMap::Empty,
            Some(base) => {
                let path_addr = if self.path.len() == 1 {
                    self.path[0].clone()
                } else {
                    Address::Path(self.path.clone())
                };
                base.get_submap(&path_addr)
            }
        };
        
        let new_value = f(current_value);
        self.set(new_value)
    }
    
    /// Update an existing value at the current address using a function that takes the raw value
    /// The function receives Some(value) if a value exists, None otherwise
    pub fn update_value<F>(self, f: F) -> ChoiceMap<T>
    where
        F: FnOnce(Option<T>) -> T,
    {
        let current_value = match &self.base {
            None => None,
            Some(base) => {
                let path_addr = if self.path.len() == 1 {
                    self.path[0].clone()
                } else {
                    Address::Path(self.path.clone())
                };
                base.get_submap(&path_addr).get_value()
            }
        };
        
        let new_value = f(current_value);
        self.set(new_value)
    }
    
    /// Returns an empty ChoiceMap (equivalent to Python's n())
    pub fn n(self) -> ChoiceMap<T> {
        ChoiceMap::Empty
    }
    
    /// Alias for set (equivalent to Python's v())
    pub fn v(self, value: T) -> ChoiceMap<T> {
        self.set(value)
    }
    
    /// Create a ChoiceMap from an iterator of (Address, T) pairs at the current path
    pub fn from_mapping(self, mapping: Vec<(Address, T)>) -> ChoiceMap<T> {
        let nested_map: ChoiceMap<T> = mapping.choice_map();
        let entry_map = nested_map.extend(&self.path);
        match self.base {
            None => entry_map,
            Some(base) => entry_map | base,
        }
    }
    
    /// Create a ChoiceMap from a HashMap at the current path (equivalent to Python's d())
    pub fn d(self, map: HashMap<String, T>) -> ChoiceMap<T> {
        let nested_map: ChoiceMap<T> = map.choice_map();
        let entry_map = nested_map.extend(&self.path);
        match self.base {
            None => entry_map,
            Some(base) => entry_map | base,
        }
    }
    
    /// Create a ChoiceMap from key-value pairs at the current path (equivalent to Python's kw())
    pub fn kw(self, pairs: Vec<(Address, T)>) -> ChoiceMap<T> {
        self.from_mapping(pairs)
    }
    
    /// Create a Switch ChoiceMap at the current path
    pub fn switch(self, idx: i32, choices: Vec<ChoiceMap<T>>) -> ChoiceMap<T> {
        let switch_map = ChoiceMap::Switch(idx, choices);
        let entry_map = switch_map.extend(&self.path);
        match self.base {
            None => entry_map,
            Some(base) => entry_map | base,
        }
    }
    
}

impl<T: Clone + Debug> ChoiceMap<T> {
    /// Create a builder for setting a value at the given path
    pub fn at(&self, path: &[Address]) -> ChoiceMapBuilder<T> {
        ChoiceMapBuilder::new(self.clone(), path.to_vec())
    }
    
    /// Convenience method to create a ChoiceMap from key-value pairs (like Python's kw)
    pub fn kw(pairs: Vec<(Address, T)>) -> Self {
        pairs.choice_map()
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::{s, sym, path};

    #[test]
    fn test_address_macros() {
        // Test symbol macro
        assert_eq!(sym!(x), Address::Symbol("x".to_string()));
        assert_eq!(sym!("hello"), Address::Symbol("hello".to_string()));

        // Test path macro
        assert_eq!(path!(x, y), Address::Path(vec![sym!(x), sym!(y)]));
        assert_eq!(path!(sym!(x), sym!(y)), Address::Path(vec![sym!(x), sym!(y)]));
        
        // Test empty path macro
        assert_eq!(path!(), Address::Path(vec![]));
    }

    #[test]
    fn test_selection() {
        // Test single symbol selection
        let selection = s!(x);
        assert!(selection == Selection::Static(Box::new(Selection::All), sym!(x)));

        // Test path selection - now creates nested structure
        let selection = s!(x, y, z);
        let expected = Selection::Static(
            Box::new(Selection::Static(
                Box::new(Selection::Static(
                    Box::new(Selection::Leaf),
                    sym!(z)
                )),
                sym!(y)
            )),
            sym!(x)
        );
        assert!(selection == expected);

        // Test union of selections
        let selection = s!(x) | s!(z, y);
        let expected_zy = Selection::Static(
            Box::new(Selection::Static(
                Box::new(Selection::Leaf),
                sym!(y)
            )),
            sym!(z)
        );
        assert!(selection == Selection::Or(
            Box::new(Selection::Static(Box::new(Selection::All), sym!(x))), 
            Box::new(expected_zy)
        ));

        assert!(selection[sym!(x)]);
        assert!(selection[path!(z, y)]);
        assert!(!selection[path!(z, y, tail)]);

        let selection = s!(x);
        assert!(selection[sym!(x)]);
        assert!(selection[path!(x)]);
        assert!(selection[path!(x, y)]);
        assert!(selection[path!(x, y, z)]);

        let selection = s!(x, y, z);
        assert!(selection[path!(x, y, z)]);
        assert!(!selection[sym!(x)]);
        assert!(!selection[path!(x, y)]);
    }


    #[test]
    fn test_wildcard_selection() {
        let selection = s!(x) | s!(*, y);

        assert!(selection[sym!(x)]);
        assert!(selection[path!(any_address, y)]);
        assert!(selection[path!(random, y, tail)]);
    }


    #[test]
    fn test_selection_all() {
        let selection = Selection::All;

        assert!(selection == !!selection.clone());
        assert!(selection[sym!(x)]);
        assert!(selection[path!(y, z)]);
        assert!(selection[path!(x, y, z)]);
    }


    #[test]
    fn test_selection_none() {
        let selection = Selection::None;

        assert!(selection == !!selection.clone());
        assert!(!selection[sym!(x)]);
        assert!(!selection[path!(y, z)]);
        
        // Test with empty path
        assert!(!selection[path!()]);
        
        // None can't be extended - it stays None
        assert!(selection.extend(&[sym!(a), sym!(b)]) == Selection::None);
    }

    #[test]
    fn test_selection_complement() {
        let selection = s!(x) | s!(y);
        let complement = !selection.clone();
        assert!(!complement[sym!(x)]);
        assert!(!complement[sym!(y)]);
        assert!(complement[sym!(z)]);

        assert!(!!selection.clone() == selection);

        assert!(!Selection::All == Selection::None);
        assert!(!Selection::None == Selection::All);
    }

    #[test]
    fn test_selection_and() {
        let sel1 = s!(x) | s!(y);
        let sel2 = s!(y) | s!(z);
        let and_sel = sel1.clone() & sel2.clone();
        assert!(!and_sel[sym!(x)]);
        assert!(and_sel[sym!(y)]);
        assert!(!and_sel.check());
        assert!(and_sel.get_subselection(&sym!(y)).check());
        assert!(!and_sel[sym!(z)]);

        // Test optimization: AllSel() & other = other
        let all_sel = Selection::All;
        assert!(all_sel.clone() & sel1.clone() == sel1);
        assert!(sel1.clone() & all_sel.clone() == sel1);

        // Test optimization: NoneSel() & other = NoneSel()
        let none_sel = Selection::None;
        assert!(none_sel.clone() & sel1.clone() == none_sel);
        assert!(sel1.clone() & none_sel.clone() == none_sel);

        // idempotence
        assert!(sel1.clone() & sel1.clone() == sel1);
        assert!(sel2.clone() & sel2.clone() == sel2);
    }

    #[test]
    fn test_selection_or() {
        let sel1 = s!(x);
        let sel2 = s!(y);
        let or_sel = sel1.clone() | sel2.clone();
        assert!(or_sel[sym!(x)]);
        assert!(or_sel[sym!(y)]);
        assert!(!or_sel[sym!(z)]);
        assert!(or_sel.get_subselection(&sym!(y)).check());

        let all_sel = Selection::All;
        assert!(all_sel.clone() | sel1.clone() == all_sel);
        assert!(sel1.clone() | all_sel.clone() == all_sel);

        let none_sel = Selection::None;
        assert!(none_sel.clone() | sel1.clone() == sel1);
        assert!(sel1.clone() | none_sel.clone() == sel1);

        // idempotence
        assert!(sel1.clone() | sel1.clone() == sel1);
        assert!(sel2.clone() | sel2.clone() == sel2);
    }

    #[test]
    fn test_selection_combination() {
        let sel1 = s!(x) | s!(y);
        let sel2 = s!(y) | s!(z);
        let combined_sel = (sel1 & sel2) | s!(w);
        assert!(!combined_sel[sym!(x)]);
        assert!(combined_sel[sym!(y)]);
        assert!(!combined_sel[sym!(z)]);
        assert!(combined_sel[sym!(w)]);
    }

    #[test]
    fn test_selection_contains() {
        let sel = s!(x) | s!(y, z);

        // Test that contains works like index
        let x = sym!(x);
        assert!(sel.contains(&x));
        assert!(sel[x]);

        let y_z = path!(y, z);
        assert!(sel.contains(&y_z));
        assert!(sel[y_z]);

        let y = sym!(y);
        assert!(!sel.contains(&y));
        assert!(!sel[y]);

        let w = sym!(w);
        assert!(!sel.contains(&w));
        assert!(!sel[w]);

        // Test with nested selections
        let nested_sel = s!(c).extend(&[sym!(a), sym!(b)]);

        assert!(nested_sel.contains(&path!(a, b, c)));
        assert!(nested_sel[path!(a, b, c)]);

        assert!(!nested_sel.contains(&path!(a, b)));
        assert!(!nested_sel[path!(a, b)]);

        // check works like contains
        assert!(!nested_sel.call(sym!(a)).call(sym!(b)).check());
        assert!(nested_sel.call(sym!(a)).call(sym!(b)).call(sym!(c)).check());
    }

    #[test]
    fn test_static_sel() {
        let xy_sel = s!(x, y);
        // Test with empty path
        assert!(!xy_sel[path!()]);
        assert!(xy_sel[path!(x, y)]);
        assert!(!xy_sel[path!(other_address)]);

        let nested_true_sel = s!(x).extend(&[sym!(y)]);
        assert!(nested_true_sel[path!(y, x)]);
        assert!(!nested_true_sel[sym!(y)]);
    }


    #[test]
    fn test_selection_extend() {
        // Test extending Selection::All
        let base_selection = Selection::All;
        let extended = base_selection.extend(&[sym!(x)]);
        
        // Should match anything under x
        assert!(extended[path!(x, y)]);
        assert!(extended[path!(x, y, z)]);
        assert!(!extended[sym!(y)]);
        
        // Test extending with multiple components
        let multi_extended = base_selection.extend(&[sym!(a), sym!(b)]);
        assert!(multi_extended[path!(a, b, anything)]);
        assert!(!multi_extended[path!(a, c)]);
        assert!(!multi_extended[sym!(a)]);
        
        // Test extending a leaf selection
        let leaf_selection = Selection::Leaf;
        let leaf_extended = leaf_selection.extend(&[sym!(x), sym!(y)]);
        assert!(leaf_extended[path!(x, y)]);
        assert!(!leaf_extended[path!(x, y, z)]);
        
        // Test extending a specific selection
        let specific = s!(z);
        let specific_extended = specific.extend(&[sym!(a), sym!(b)]);
        assert!(specific_extended[path!(a, b, z, anything)]);
        assert!(!specific_extended[path!(a, b, w)]);
    }

    
    #[test]
    fn test_empty() {
        let cm = ChoiceMap::<i32>::Empty;
        assert!(cm.is_empty());
    }

    #[test]
    fn test_choice() {
        let cm = ChoiceMap::Choice(42);
        assert!(cm.get_value() == Some(42));
        assert!(!cm.is_empty());

        assert!(cm.contains(&path!()));

        // Choice with a mask that is concrete False is empty.
        let mask: ChoiceMap<f64> = Mask::new(42.0, false).choice_map();
        assert!(mask.is_empty());

        // Masks with concrete `True` flags have their masks stripped off
        let mask: ChoiceMap<f64> = Mask::new(42.0, true).choice_map();
        assert!(mask == ChoiceMap::Choice(42.0));
    }

    #[test]
    fn test_hash_map_into_choice_map() {
        let mut map: HashMap<String, i32> = HashMap::new();
        map.insert("x".to_string(), 1);
        map.insert("y".to_string(), 2);
        let cm: ChoiceMap<i32> = map.choice_map();

        assert_eq!(cm.get(&sym!(x)), 1);
        assert_eq!(cm.get(&sym!(y)), 2);
        
        // This will panic as expected
        assert!(cm.contains(&sym!(x)));
        assert!(cm.contains(&sym!(y)));
        assert!(!cm.contains(&sym!(z)));
    }

    #[test]
    fn test_mapping_into_choice_map() {
        let mapping = vec![
            (sym!(x), 1),
            (path!(y, z), 2),
            (path!(w, v, u), 3),
        ];
        let cm: ChoiceMap<i32> = mapping.choice_map();

        assert_eq!(cm.get(&sym!(x)), 1);
        assert_eq!(cm.get(&path!(y, z)), 2);
        assert_eq!(cm.get(&path!(w, v, u)), 3);

        assert!(cm.contains(&sym!(x)));
        assert!(cm.contains(&path!(y, z)));
        assert!(cm.contains(&path!(w, v, u)));
    }


    #[test]
    fn test_extend_through_at() {
        // Create an initial ChoiceMap
        let initial_chm = ChoiceMap::kw(vec![
            (sym!(x), 1),
            (path!(y, z), 2),
        ]);

        // Extend the ChoiceMap using 'at'
        let extended_chm = initial_chm.at(&[sym!(y), sym!(w)]).set(3);

        // Test that the original values are preserved
        assert_eq!(extended_chm.get(&sym!(x)), 1);
        assert_eq!(extended_chm.get(&path!(y, z)), 2);

        // Test that the new value is correctly set
        assert_eq!(extended_chm.get(&path!(y, w)), 3);

        // Test that we can chain multiple extensions
        let multi_extended_chm = initial_chm
            .at(&[sym!(y), sym!(w)]).set(3)
            .at(&[sym!(a), sym!(b), sym!(c)]).set(4);

        assert_eq!(multi_extended_chm.get(&sym!(x)), 1);
        assert_eq!(multi_extended_chm.get(&path!(y, z)), 2);
        assert_eq!(multi_extended_chm.get(&path!(y, w)), 3);
        assert_eq!(multi_extended_chm.get(&path!(a, b, c)), 4);

        // Test overwriting an existing value
        let overwritten_chm = initial_chm.at(&[sym!(y), sym!(z)]).set(5);

        assert_eq!(overwritten_chm.get(&sym!(x)), 1);
        assert_eq!(overwritten_chm.get(&path!(y, z)), 5); // Value has been overwritten
    }


    #[test]
    fn test_filter() {
        let chm = ChoiceMap::kw(vec![
            (sym!(x), 1),
            (sym!(y), 2),
            (sym!(z), 3),
        ]);
        let sel = s!(x) | s!(y);
        let filtered = chm.filter(&sel);
        assert_eq!(filtered.get(&sym!(x)), 1);
        assert_eq!(filtered.get(&sym!(y)), 2);
        assert!(!filtered.contains(&sym!(z)));
    }

    #[test]
    fn test_mask() {
        let chm = ChoiceMap::kw(vec![
            (sym!(x), 1),
            (sym!(y), 2),
        ]);
        let masked_true = chm.mask(true);
        assert!(masked_true == chm);
        let masked_false = chm.mask(false);
        assert!(masked_false.is_empty());
    }
    
    #[test]
    fn test_extend() {
        let chm = ChoiceMap::Choice(1);
        let extended = chm.extend(&[sym!(a), sym!(b)]);
        assert_eq!(extended.get(&path!(a, b)), 1);
        assert!(extended.get_value() == None);
        assert!(extended.get_submap(&path!(a, b)).get_value() == Some(1));
    }

    #[test]
    fn test_switch_chm() {
        let chm1 = ChoiceMap::kw(vec![
            (sym!(x), 1),
            (sym!(y), 2),
        ]);
        let chm2 = ChoiceMap::kw(vec![
            (sym!(a), 3),
            (sym!(b), 4),
        ]);
        let chm3 = ChoiceMap::kw(vec![
            (sym!(p), 5),
            (sym!(q), 6),
        ]);
        let switched = ChoiceMap::switch(1, vec![chm1.clone(), chm2.clone(), chm3.clone()]);
        assert!(switched == chm2);
    }
}
