use std::fmt::{Debug, Display};

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
                write!(
                    f,
                    "[{}]",
                    path.iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
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
            Address::Path(components) => components.iter().all(|c| c.is_static()),
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
    Or(Box<Selection>, Box<Selection>),
}

impl Selection {
    pub fn check(&self) -> bool {
        match self {
            Selection::All => true,
            Selection::None => false,
            Selection::Leaf => true,
            Selection::Static(_, _) => false, // ← This should be false!
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
            single_component => self.get_subselection(&single_component),
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
            Selection::Complement(inner) => !inner.call(addr.clone()),
            Selection::Static(inner, target) => match target {
                Address::Wildcard => *inner.clone(),
                _ if target == addr => *inner.clone(),
                _ => Selection::None,
            },
            Selection::And(left, right) => left.call(addr.clone()) & right.call(addr.clone()),
            Selection::Or(left, right) => left.call(addr.clone()) | right.call(addr.clone()),
        }
    }
}

// From implementations for easy construction of selections
impl From<&[Address]> for Selection {
    fn from(addresses: &[Address]) -> Self {
        if addresses.is_empty() {
            Selection::None
        } else {
            addresses
                .iter()
                .map(|addr| {
                    match addr {
                        // For path addresses, create nested static selections
                        Address::Path(components) => {
                            if components.is_empty() {
                                Selection::None
                            } else {
                                // Create nested structure like s!(a, b, c)
                                let mut result = Selection::Leaf;
                                for component in components.iter().rev() {
                                    result = Selection::Static(Box::new(result), component.clone());
                                }
                                result
                            }
                        }
                        // For single addresses, use Selection::All as inner
                        single => Selection::Static(Box::new(Selection::All), single.clone()),
                    }
                })
                .fold(Selection::None, |acc, sel| acc | sel)
        }
    }
}

impl From<Vec<Address>> for Selection {
    fn from(addresses: Vec<Address>) -> Self {
        Selection::from(addresses.as_slice())
    }
}

impl From<&Vec<Address>> for Selection {
    fn from(addresses: &Vec<Address>) -> Self {
        Selection::from(addresses.as_slice())
    }
}

impl From<Address> for Selection {
    fn from(address: Address) -> Self {
        match address {
            // For path addresses, create nested static selections
            Address::Path(components) => {
                if components.is_empty() {
                    Selection::None
                } else {
                    // Create nested structure like s!(a, b, c)
                    let mut result = Selection::Leaf;
                    for component in components.iter().rev() {
                        result = Selection::Static(Box::new(result), component.clone());
                    }
                    result
                }
            }
            // For single addresses, use Selection::All as inner
            single => Selection::Static(Box::new(Selection::All), single),
        }
    }
}

impl From<&Address> for Selection {
    fn from(address: &Address) -> Self {
        Selection::from(address.clone())
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
        $crate::address::Selection::Static(Box::new($crate::address::Selection::All), $crate::address::Address::Symbol(stringify!($x).to_string()))
    };
    (*) => {
        $crate::address::Selection::Static($crate::address::Address::Wildcard)
    };
    ($($x:ident),+ $(,)?) => {{
        let mut result = $crate::address::Selection::Leaf;
        let components = vec![$($crate::address::Address::Symbol(stringify!($x).to_string())),+];
        for addr in components.into_iter().rev() {
            result = $crate::address::Selection::Static(Box::new(result), addr);
        }
        result
    }};
    (*, $($x:ident),+ $(,)?) => {{
        let mut result = $crate::address::Selection::All;
        let mut components = vec![$($crate::address::Address::Symbol(stringify!($x).to_string())),+];
        components.insert(0, $crate::address::Address::Wildcard);
        for addr in components.into_iter().rev() {
            result = $crate::address::Selection::Static(Box::new(result), addr);
        }
        result
    }};
}

/// Macro to create symbol addresses
#[macro_export]
macro_rules! sym {
    ($x:ident) => {
        $crate::address::Address::Symbol(stringify!($x).to_string())
    };
    ($x:expr) => {
        $crate::address::Address::Symbol($x.to_string())
    };
}

/// Macro to create path addresses
#[macro_export]
macro_rules! path {
    () => {
        $crate::address::Address::Path(vec![])
    };
    ($($x:ident),+ $(,)?) => {
        $crate::address::Address::Path(vec![$(sym!($x)),+])
    };
    ($($x:expr),+ $(,)?) => {
        $crate::address::Address::Path(vec![$($x),+])
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{path, s, sym};

    #[test]
    fn test_address_macros() {
        // Test symbol macro
        assert_eq!(sym!(x), Address::Symbol("x".to_string()));
        assert_eq!(sym!("hello"), Address::Symbol("hello".to_string()));

        // Test path macro
        assert_eq!(path!(x, y), Address::Path(vec![sym!(x), sym!(y)]));
        assert_eq!(
            path!(sym!(x), sym!(y)),
            Address::Path(vec![sym!(x), sym!(y)])
        );

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
                Box::new(Selection::Static(Box::new(Selection::Leaf), sym!(z))),
                sym!(y),
            )),
            sym!(x),
        );
        assert!(selection == expected);

        // Test union of selections
        let selection = s!(x) | s!(z, y);
        let expected_zy = Selection::Static(
            Box::new(Selection::Static(Box::new(Selection::Leaf), sym!(y))),
            sym!(z),
        );
        assert!(
            selection
                == Selection::Or(
                    Box::new(Selection::Static(Box::new(Selection::All), sym!(x))),
                    Box::new(expected_zy)
                )
        );

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
    fn test_selection_from_addresses() {
        // Test From<&[Address]>
        let addresses = [sym!(x), sym!(y), sym!(z)];
        let selection = Selection::from(&addresses[..]);
        assert!(selection[sym!(x)]);
        assert!(selection[sym!(y)]);
        assert!(selection[sym!(z)]);
        assert!(!selection[sym!(w)]);

        // Test From<Vec<Address>>
        let addresses_vec = vec![sym!(a), sym!(b)];
        let selection = Selection::from(addresses_vec);
        assert!(selection[sym!(a)]);
        assert!(selection[sym!(b)]);
        assert!(!selection[sym!(c)]);

        // Test From<&Vec<Address>>
        let addresses_vec = vec![sym!(p), sym!(q)];
        let selection = Selection::from(&addresses_vec);
        assert!(selection[sym!(p)]);
        assert!(selection[sym!(q)]);
        assert!(!selection[sym!(r)]);

        // Test From<Address>
        let selection = Selection::from(sym!(single));
        assert!(selection[sym!(single)]);
        assert!(selection[path!(single, child)]);
        assert!(!selection[sym!(other)]);

        // Test From<&Address>
        let addr = sym!(reference);
        let selection = Selection::from(&addr);
        assert!(selection[sym!(reference)]);
        assert!(selection[path!(reference, child)]);
        assert!(!selection[sym!(other)]);

        // Test empty slice
        let empty_addresses: &[Address] = &[];
        let selection = Selection::from(empty_addresses);
        assert_eq!(selection, Selection::None);
        assert!(!selection[sym!(anything)]);

        // Test with Path addresses - now fixed!
        let path_addr = path!(a, b);
        let selection = Selection::from(path_addr.clone());

        // The exact path should match
        assert!(selection[path_addr.clone()]);

        // But children should NOT match because we use Leaf, not All
        assert!(!selection[path!(a, b, child)]);

        // Test with multiple addresses including paths
        let path_addresses = vec![path!(a, b), sym!(c), Address::Index(42)];
        let selection = Selection::from(path_addresses);

        // Test each address individually
        assert!(selection[path!(a, b)]);
        assert!(selection[sym!(c)]);
        assert!(selection[Address::Index(42)]);

        // Path matches exactly but not children (due to Leaf)
        assert!(!selection[path!(a, b, child)]);

        // Symbol matches and its children (due to All)
        assert!(selection[path!(c, child)]);

        // Test non-matching addresses
        assert!(!selection[path!(a, c)]);
        assert!(!selection[sym!(d)]);
    }
}
