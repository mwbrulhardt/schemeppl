use std::convert::TryFrom;

use crate::dsl::ast::{Expression, Literal};

impl TryFrom<lexpr::Value> for Expression {
    type Error = String;

    fn try_from(v: lexpr::Value) -> Result<Self, Self::Error> {
        use lexpr::Value::*;

        Ok(match v {
            Bool(b) => Expression::Constant(Literal::Boolean(b)),

            Number(n) => {
                if n.is_f64() {
                    Expression::Constant(Literal::Float(n.as_f64().unwrap()))
                } else {
                    Expression::Constant(Literal::Integer(n.as_i64().unwrap()))
                }
            }
            String(s) => Expression::Constant(Literal::String(s.into())),
            Symbol(s) => Expression::Variable(s.into()),

            Cons(pair) => {
                // Collect the list (and an eventual improper tail) into Vec<Expression>
                let mut elems = Vec::new();
                for (car, maybe_tail) in pair.into_iter() {
                    elems.push(Expression::try_from(car).unwrap());

                    if let Some(tail) = maybe_tail {
                        if !matches!(tail, lexpr::Value::Nil | lexpr::Value::Null) {
                            elems.push(Expression::try_from(tail).unwrap());
                        }
                    }
                }

                // special forms -----------------------------------------------------------
                match &elems[..] {
                    // (quote x)
                    [Expression::Variable(k), x] if k == "quote" => {
                        Expression::Quote(Box::new(x.clone()))
                    }

                    // (lambda (args...) body...)
                    [Expression::Variable(k), Expression::List(arg_elems), body @ ..]
                        if k == "lambda" =>
                    {
                        let args = arg_elems
                            .iter()
                            .map(|e| match e {
                                Expression::Variable(name) => Ok(name.clone()),
                                _ => Err("lambda arguments must be symbols"),
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        let body_expr = if body.len() == 1 {
                            body[0].clone()
                        } else {
                            Expression::List(body.to_vec())
                        };
                        Expression::Lambda(args, Box::new(body_expr))
                    }

                    // (if test then else)
                    [Expression::Variable(k), test, then_, else_] if k == "if" => Expression::If(
                        Box::new(test.clone()),
                        Box::new(then_.clone()),
                        Box::new(else_.clone()),
                    ),

                    // (define name expr)
                    [Expression::Variable(k), Expression::Variable(name), expr]
                        if k == "define" =>
                    {
                        Expression::Define(name.clone(), Box::new(expr.clone()))
                    }

                    // (sample name dist)
                    [Expression::Variable(k), name, dist] if k == "sample" => {
                        // Either a string expression or a function call that evaluates to a string
                        let name = match name {
                            Expression::Variable(symbol) => {
                                Expression::Constant(Literal::String(symbol.clone()))
                            }
                            _ => name.clone(),
                        };

                        Expression::Sample {
                            distribution: Box::new(dist.clone()),
                            name: name.into(),
                        }
                    }

                    // (observe name dist value)
                    [Expression::Variable(k), name, dist, expr] if k == "observe" => {
                        // Either a string expression or a function call that evaluates to a string
                        let name = match name {
                            Expression::Variable(symbol) => {
                                Expression::Constant(Literal::String(symbol.clone()))
                            }
                            _ => name.clone(),
                        };

                        Expression::Observe {
                            name: Box::new(name.clone()),
                            distribution: Box::new(dist.clone()),
                            observed: Box::new(expr.clone()),
                        }
                    }

                    // (for-each proc seq)
                    [Expression::Variable(k), proc, seq] if k == "for-each" => {
                        Expression::ForEach {
                            func: Box::new(proc.clone()),
                            seq: Box::new(seq.clone()),
                        }
                    }

                    // anything else → ordinary list
                    _ => Expression::List(elems),
                }
            }

            Nil | Null => Expression::List(vec![]),

            other => return Err(format!("unhandled lexpr value: {:?}", other)),
        })
    }
}

pub fn parse_string(input: &str) -> Vec<Expression> {
    // wrap in an extra parens so we get all top‐level forms at once:
    let v = lexpr::from_str(input).unwrap_or_else(|e| panic!("lexpr parsing error: {}", e));
    if let lexpr::Value::Cons(pair) = v {
        // convert each car in turn
        let mut out = Vec::new();
        let mut rest = Some((pair.car().clone(), pair.cdr().clone()));
        while let Some((h, t)) = rest.take() {
            out.push(Expression::try_from(h).unwrap());
            rest = match t {
                lexpr::Value::Cons(pair2) => Some((pair2.car().clone(), pair2.cdr().clone())),
                lexpr::Value::Nil => None,
                lexpr::Value::Null => None,
                other => {
                    out.push(Expression::try_from(other).unwrap());
                    None
                }
            };
        }
        out
    } else {
        panic!("expected wrapped list");
    }
}

#[macro_export]
macro_rules! gen {
    // Version that takes parameter names and creates a SchemeGenerativeFunction
    ([$($param:ident),*] { $($body:tt)* }) => {{
        let src = stringify!($($body)*);
        let wrapped = format!("( {} )", src);
        let exprs = $crate::dsl::parser::parse_string(&wrapped);
        let param_names = vec![$(stringify!($param).to_string()),*];
        $crate::dynamic::trace::SchemeGenerativeFunction::new(exprs, param_names)
    }};

    // Original version that just returns expressions (for backward compatibility)
    ( $($tt:tt)* ) => {{
        let src = stringify!($($tt)*);
        let wrapped = format!("( {} )", src);
        let exprs = $crate::dsl::parser::parse_string(&wrapped);
        exprs
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixture() {
        let exprs = gen!(
            (mixture (list (normal mu1 1.0) (normal mu2 1.0)) (list p (- 1.0 p)))
        );

        let mixture = Expression::List(vec![
            Expression::Variable("mixture".into()),
            Expression::List(vec![
                Expression::Variable("list".into()),
                Expression::List(vec![
                    Expression::Variable("normal".into()),
                    Expression::Variable("mu1".into()),
                    Expression::Constant(Literal::Float(1.0)),
                ]),
                Expression::List(vec![
                    Expression::Variable("normal".into()),
                    Expression::Variable("mu2".into()),
                    Expression::Constant(Literal::Float(1.0)),
                ]),
            ]),
            Expression::List(vec![
                Expression::Variable("list".into()),
                Expression::Variable("p".into()),
                Expression::List(vec![
                    Expression::Variable("-".into()),
                    Expression::Constant(Literal::Float(1.0)),
                    Expression::Variable("p".into()),
                ]),
            ]),
        ]);

        assert_eq!(exprs[0], mixture);
    }

    #[test]
    fn test_sample() {
        let exprs = gen!(
            (mixture (list (normal mu1 1.0) (normal mu2 1.0)) (list p (- 1.0 p)))
        );

        let mixture = Expression::List(vec![
            Expression::Variable("mixture".into()),
            Expression::List(vec![
                Expression::Variable("list".into()),
                Expression::List(vec![
                    Expression::Variable("normal".into()),
                    Expression::Variable("mu1".into()),
                    Expression::Constant(Literal::Float(1.0)),
                ]),
                Expression::List(vec![
                    Expression::Variable("normal".into()),
                    Expression::Variable("mu2".into()),
                    Expression::Constant(Literal::Float(1.0)),
                ]),
            ]),
            Expression::List(vec![
                Expression::Variable("list".into()),
                Expression::Variable("p".into()),
                Expression::List(vec![
                    Expression::Variable("-".into()),
                    Expression::Constant(Literal::Float(1.0)),
                    Expression::Variable("p".into()),
                ]),
            ]),
        ]);

        assert_eq!(exprs[0], mixture);
    }
}
