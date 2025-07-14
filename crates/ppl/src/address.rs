use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter, Result};
use std::ops::{BitOr, BitXor, Not};

// Address can be single component or multiple components (tuple)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Address {
    Symbol(String),
    Path(Vec<String>),
}

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

// Display implementations for debugging
impl Display for Address {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Address::Symbol(s) => write!(f, "{}", s),
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
        }
    }
}

/// Selection types
#[derive(Clone, PartialEq, Debug)]
pub enum Selection {
    All,
    None,
    Str(String),
    Tuple(Vec<String>),
    Dict(HashMap<String, Selection>),
    Complement(Box<Selection>),
    In(Box<Selection>, Box<Selection>),
    Or(Box<Selection>, Box<Selection>),
}

impl Selection {
    pub fn check(&self, addr: Address) -> (bool, Selection) {
        match self {
            Selection::All => (true, self.clone()),
            Selection::None => (false, self.clone()),
            Selection::Str(s) => {
                let check = addr == Address::Symbol(s.clone());
                let result = if check {
                    Selection::All
                } else {
                    Selection::None
                };
                (check, result)
            }
            Selection::Tuple(t) => match t.len() {
                0 => (false, Selection::None),
                1 => {
                    let check = addr == Address::Symbol(t[0].clone());
                    let result = if check {
                        Selection::All
                    } else {
                        Selection::None
                    };
                    (check, result)
                }
                _ => {
                    let check = addr == Address::Symbol(t[0].clone());
                    if check {
                        let remaining = t[1..].to_vec();
                        let result = Selection::Tuple(remaining);
                        (true, result)
                    } else {
                        (false, Selection::None)
                    }
                }
            },
            Selection::Dict(map) => {
                // Check that the address is a symbol
                match addr {
                    Address::Symbol(s) => {
                        let check = map.contains_key(&s);
                        (
                            check,
                            if check {
                                map.get(&s).unwrap().clone()
                            } else {
                                Selection::None
                            },
                        )
                    }
                    _ => (false, Selection::None),
                }
            }
            Selection::Complement(inner) => {
                let (check, rest) = inner.check(addr.clone());
                (!check, Selection::Complement(Box::new(rest)))
            }
            Selection::In(left, right) => {
                let (check1, r1) = left.check(addr.clone());
                let (check2, r2) = right.check(addr.clone());
                (check1 && check2, Selection::In(Box::new(r1), Box::new(r2)))
            }
            Selection::Or(left, right) => {
                let (check1, r1) = left.check(addr.clone());
                let (check2, r2) = right.check(addr.clone());
                (check1 || check2, Selection::Or(Box::new(r1), Box::new(r2)))
            }
        }
    }

    /// Check if an address is contained in this selection
    pub fn contains(&self, addr: &Address) -> bool {
        let (check, _) = self.check(addr.clone());
        check
    }
}

impl BitXor for Selection {
    type Output = Selection;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Selection::In(Box::new(self), Box::new(rhs))
    }
}

impl BitOr for Selection {
    type Output = Selection;

    fn bitor(self, rhs: Self) -> Self::Output {
        Selection::Or(Box::new(self), Box::new(rhs))
    }
}

impl Not for Selection {
    type Output = Selection;

    fn not(self) -> Self::Output {
        Selection::Complement(Box::new(self))
    }
}

/// Macro to create selections
#[macro_export]
macro_rules! select {
    // Handle empty invocation: select!()
    () => {
        $crate::address::Selection::None
    };
    // Handle None: select!(None)
    (None) => {
        $crate::address::Selection::None
    };
    // Handle empty tuple (all): select!(())
    (()) => {
        $crate::address::Selection::All
    };
    // Handle single identifier: select!(x)
    ($x:ident) => {
        $crate::address::Selection::Str(stringify!($x).to_string())
    };
    // Handle single string literal: select!("x")
    ($x:literal) => {
        $crate::address::Selection::Str($x.to_string())
    };
    // Handle tuple of identifiers: select!((x, y))
    (($($x:ident),+ $(,)?)) => {
        $crate::address::Selection::Tuple(vec![$(stringify!($x).to_string()),+])
    };
    // Handle tuple of string literals: select!(("x", "y"))
    (($($x:literal),+ $(,)?)) => {
        $crate::address::Selection::Tuple(vec![$($x.to_string()),+])
    };
    // Handle dictionary syntax: select!({ key: value, ... })
    ({ $($key:ident: $value:tt),+ $(,)? }) => {
        {
            let mut map = std::collections::HashMap::new();
            $(
                map.insert(stringify!($key).to_string(), select!($value));
            )+
            $crate::address::Selection::Dict(map)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::select;

    #[test]
    fn test_select_macro() {
        // Handle empty invocation
        let expected = Selection::None;
        let s = select!();
        assert_eq!(s, expected);

        // Handle None
        let expected = Selection::None;
        let s = select!(None);
        assert_eq!(s, expected);

        // Handle empty tuple (all)
        let expected = Selection::All;
        let s = select!(());
        assert_eq!(s, expected);

        // Handle single identifier
        let expected = Selection::Str("x".to_string());
        let s = select!(x);
        assert_eq!(s, expected);

        let s = select!("x");
        assert_eq!(s, expected);

        // Handle hierarchical address
        let expected = Selection::Tuple(vec!["outer".to_string(), "inner".to_string()]);
        let s = select!((outer, inner));
        assert_eq!(s, expected);

        let s = select!(("outer", "inner"));
        assert_eq!(s, expected);

        // Handle dictionary of key-value pairs
        let expected = Selection::Dict(HashMap::from([(
            "outer".to_string(),
            Selection::Str("inner".to_string()),
        )]));
        let s = select!({
            outer: inner,
        });
        assert_eq!(s, expected);
    }

    #[test]
    fn test_selection_check_str() {
        // Test single symbol selection
        let s = select!(x);

        let (check, result) = s.check("x".into());
        assert!(check);
        assert_eq!(result, select!(()));

        let (check, result) = s.check("y".into());
        assert!(!check);
        assert_eq!(result, select!(None));
    }

    #[test]
    fn test_selection_check_all() {
        let s = select!(());

        // All should match any address
        let (check, result) = s.check("x".into());
        assert!(check);
        assert_eq!(result, select!(()));

        let (check, result) = s.check("anything".into());
        assert!(check);
        assert_eq!(result, select!(()));

        let (check, result) = s.check(Address::Path(vec!["a".to_string(), "b".to_string()]));
        assert!(check);
        assert_eq!(result, select!(()));
    }

    #[test]
    fn test_selection_check_none() {
        let s = select!(None);

        // None should never match any address
        let (check, result) = s.check("x".into());
        assert!(!check);
        assert_eq!(result, select!(None));

        let (check, result) = s.check("anything".into());
        assert!(!check);
        assert_eq!(result, select!(None));

        let (check, result) = s.check(Address::Path(vec!["a".to_string(), "b".to_string()]));
        assert!(!check);
        assert_eq!(result, select!(None));
    }

    #[test]
    fn test_selection_check_tuple() {
        // Test empty tuple
        let selection = Selection::Tuple(vec![]);
        let (check, result) = selection.check("x".into());
        assert!(!check);
        assert_eq!(result, Selection::None);

        // Test single element tuple - now fixed!
        let selection = Selection::Tuple(vec!["x".to_string()]);
        let (check, result) = selection.check("x".into());
        assert!(check);
        assert_eq!(result, Selection::All);

        // Non-matching should now correctly return false
        let (check, result) = selection.check("y".into());
        assert!(!check); // Fixed: now correctly returns false
        assert_eq!(result, Selection::None);

        // Test multi-element tuple
        let selection = Selection::Tuple(vec!["x".to_string(), "y".to_string(), "z".to_string()]);

        // First element should match and return remaining tuple
        let (check, result) = selection.check("x".into());
        assert!(check);
        assert_eq!(
            result,
            Selection::Tuple(vec!["y".to_string(), "z".to_string()])
        );

        // Non-matching element should fail
        let (check, result) = selection.check("a".into());
        assert!(!check);
        assert_eq!(result, Selection::None);
    }

    #[test]
    fn test_selection_check_dict() {
        let s = select!({ x: inner, y: () });

        // Test matching key with string selection
        let (check, result) = s.check("x".into());
        assert!(check);
        assert_eq!(result, select!(inner));

        // Test matching key with All selection
        let (check, result) = s.check("y".into());
        assert!(check);
        assert_eq!(result, select!(()));

        // Test non-matching key
        let (check, result) = s.check("z".into());
        assert!(!check);
        assert_eq!(result, select!(None));

        // Test with Path address (should fail since Dict expects Symbol)
        let (check, result) = s.check(Address::Path(vec!["x".to_string(), "y".to_string()]));
        assert!(!check);
        assert_eq!(result, select!(None));
    }

    #[test]
    fn test_selection_check_complement() {
        let s = !select!(x);

        // Should NOT match what the inner selection matches
        let (check, result) = s.check("x".into());
        assert!(!check);
        assert_eq!(result, !select!(()));

        // Should match what the inner selection doesn't match
        let (check, result) = s.check("y".into());
        assert!(check);
        assert_eq!(result, !select!(None));
    }

    #[test]
    fn test_selection_check_in() {
        let left = select!(x);
        let right = select!(());
        let s = Selection::In(Box::new(left), Box::new(right));

        // Should match only if BOTH left and right match
        let (check, result) = s.check("x".into());
        assert!(check);
        assert_eq!(
            result,
            Selection::In(Box::new(Selection::All), Box::new(Selection::All))
        );

        // Should fail if left doesn't match (even though right would)
        let (check, result) = s.check("y".into());
        assert!(!check);
        assert_eq!(
            result,
            Selection::In(Box::new(Selection::None), Box::new(Selection::All))
        );
    }

    #[test]
    fn test_selection_check_or() {
        let s = select!(x) | select!(y);

        // Should match if left matches
        let (check, result) = s.check("x".into());
        assert!(check);
        assert_eq!(result, select!(()) | select!(None));

        // Should match if right matches
        let (check, result) = s.check("y".into());
        assert!(check);
        assert_eq!(result, select!(None) | select!(()));

        // Should fail if neither matches
        let (check, result) = s.check("z".into());
        assert!(!check);
        assert_eq!(result, select!(None) | select!(None));
    }

    #[test]
    fn test_selection_contains() {
        // Test that contains method works correctly
        let s = select!(x);
        assert!(s.contains(&"x".into()));
        assert!(!s.contains(&"y".into()));

        // Test with compound selection
        let s = select!(x) | select!(y);

        assert!(s.contains(&"x".into()));
        assert!(s.contains(&"y".into()));
        assert!(!s.contains(&"z".into()));
    }
}
