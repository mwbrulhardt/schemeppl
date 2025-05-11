use crate::{Expression, Literal};
use std::convert::TryFrom;

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
                    [Expression::Variable(k), Expression::Variable(name), dist]
                        if k == "sample" =>
                    {
                        Expression::Sample {
                            distribution: Box::new(dist.clone()),
                            name: name.into(),
                        }
                    }

                    // (observe name dist value)
                    [Expression::Variable(k), name, dist, expr] if k == "observe" => {
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
    ( $($tt:tt)* ) => {{
        // turn the tokens back into a string like "#t x (lambda (x) x) (* x x)"
        let src = stringify!($($tt)*);
        // wrap in parens so our parser sees one top-level list of forms
        let wrapped = format!("( {} )", src);
        // hand off to your parser
        let exprs = $crate::parser::parse_string(&wrapped);

        exprs
    }};
}

#[cfg(test)]
mod tests {
    use rand::distributions::{Bernoulli, Distribution};
    use statrs::distribution::Normal;
    use std::collections::HashMap;
    use std::collections::HashSet;

    use super::*;
    use crate::{mh, GenerativeFunction, Value};

    #[test]
    fn test_define() {
        let exprs = gen!(
            (define x 5.0)
            (define y 6.0)
        );

        println!("Expr: {:?}", exprs);

        //assert!(Expression::Define("x".into(), ()))
    }

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

        println!("Expr: {:?}", exprs);
    }

    #[test]
    fn test_for_each() {
        let data = vec![
            0.89114093,
            0.264701206,
            1.229336011,
            0.781903503,
            0.442450484,
            -2.007232581,
            1.920186601,
            1.657035609,
            1.131176314,
            1.532964528,
            -1.959504631,
        ];
        let wrapped_data = Value::List(data.into_iter().map(|x| Value::Float(x)).collect());

        let model = gen!(
            (sample mu (normal 0.0 1.0))

            (define observe-point (lambda (x) (observe (gensym) (normal mu 1.0) x)))

            (for-each observe-point data)
        );

        println!("Model: {:?}", model);

        let program = GenerativeFunction::new(model, vec!["data".to_string()], HashMap::new(), 42);

        let trace = program.simulate(vec![wrapped_data]).unwrap();

        println!("Trace: {:?}", trace);
    }

    #[test]
    fn test_model() {
        let n = 1000;

        let model = gen!(
            // Priors
            (sample mu1 (normal 0.0 1.0))
            (sample mu2 (normal 0.0 1.0))

            // Ordering
            (condition (< mu1 mu2))

            // Mixture
            (define p 0.5)
            (define mix (mixture (list (normal mu1 1.0) (normal mu2 1.0)) (list p (- 1.0 p))))

            // Observe
            (observe x0 mix 0.8911409301893696)
            (observe x1 mix 0.26470120566977173)
            (observe x2 mix 1.229336011002809)
            (observe x3 mix 0.7819035026306239)
            (observe x4 mix 0.4424504842353567)
            (observe x5 mix -2.0072325809871394)
            (observe x6 mix 1.9201866009058568)
            (observe x7 mix 1.6570356094582315)
            (observe x8 mix 1.1311763142057991)
            (observe x9 mix 1.5329645280322546)
            (observe x10 mix -1.959504631150351)
            (observe x11 mix 1.7916399692727243)
            (observe x12 mix 0.25295070586188895)
            (observe x13 mix -0.26050942883393813)
            (observe x14 mix -1.4340629400942282)
            (observe x15 mix 2.1598918958571875)
            (observe x16 mix -2.0489432234322416)
            (observe x17 mix -2.371840294481374)
            (observe x18 mix 0.5340133288604277)
            (observe x19 mix -1.7994000510144301)
            (observe x20 mix -1.118065554305085)
            (observe x21 mix 1.3627945363108405)
            (observe x22 mix 0.052278859087714125)
            (observe x23 mix 0.4456657517099658)
            (observe x24 mix 1.9657257504350474)
            (observe x25 mix 1.4265637116035415)
            (observe x26 mix -2.096160389397204)
            (observe x27 mix 0.23720577604578819)
            (observe x28 mix -0.28805101753714424)
            (observe x29 mix 0.1634410642403108)
            (observe x30 mix 1.362658716200644)
            (observe x31 mix 1.0458770931684085)
            (observe x32 mix 2.109328929187112)
            (observe x33 mix -0.7516821642853673)
            (observe x34 mix -0.428746686083353)
            (observe x35 mix -0.3227081823752269)
            (observe x36 mix 1.8187866151332877)
            (observe x37 mix 0.17573816169679035)
            (observe x38 mix 0.28995652332642474)
            (observe x39 mix 0.8670875335623385)
            (observe x40 mix 2.7708747075367457)
            (observe x41 mix 2.4959472672500205)
            (observe x42 mix 1.963981956001585)
            (observe x43 mix 0.9318513791404165)
            (observe x44 mix -1.410365396287913)
            (observe x45 mix -0.5377996152456757)
            (observe x46 mix -1.1291130451339755)
            (observe x47 mix 2.053778075838431)
            (observe x48 mix -0.6107267225211761)
            (observe x49 mix 1.6109205309967018)
            (observe x50 mix -0.7445301217159537)
            (observe x51 mix -2.292449353682473)
            (observe x52 mix -1.5486684678485771)
            (observe x53 mix -3.104190437289443)
            (observe x54 mix 0.7928397543082162)
            (observe x55 mix 2.162591945601675)
            (observe x56 mix -1.352166278739354)
            (observe x57 mix -3.100093867295413)
            (observe x58 mix -2.0652106264384598)
            (observe x59 mix 1.3506324925860667)
            (observe x60 mix 3.6154775381360333)
            (observe x61 mix -0.6309845380780119)
            (observe x62 mix 1.9707402876566589)
            (observe x63 mix -1.0001452611494965)
            (observe x64 mix 1.5623870678122578)
            (observe x65 mix 0.3761823269192398)
            (observe x66 mix -1.6251100352176493)
            (observe x67 mix 0.46040148066309494)
            (observe x68 mix 2.55581889724124)
            (observe x69 mix -0.8534826343032852)
            (observe x70 mix 2.4193787041281967)
            (observe x71 mix 1.4661367482675183)
            (observe x72 mix -0.1091047501487008)
            (observe x73 mix -0.931287234006658)
            (observe x74 mix -0.8313285995835573)
            (observe x75 mix -0.5159992108132501)
            (observe x76 mix -1.2967927610956922)
            (observe x77 mix -2.4877432032712807)
            (observe x78 mix 1.3474279150321717)
            (observe x79 mix -1.1315852340292247)
            (observe x80 mix -1.0673318987662117)
            (observe x81 mix -0.7476211989545432)
            (observe x82 mix -2.2672771386399635)
            (observe x83 mix -1.2279502133550007)
            (observe x84 mix 0.664546378076192)
            (observe x85 mix 1.178525136364507)
            (observe x86 mix 0.8571184939291463)
            (observe x87 mix 0.2862379924495405)
            (observe x88 mix -0.4552839601906412)
            (observe x89 mix 1.1517472549891883)
            (observe x90 mix -1.6440064969708457)
            (observe x91 mix 1.2580553257701756)
            (observe x92 mix -0.16287371888685853)
            (observe x93 mix -2.4961073317413867)
            (observe x94 mix 1.5377908532847493)
            (observe x95 mix -0.6259144957229507)
            (observe x96 mix -1.5877761800831287)
            (observe x97 mix 0.22031835036620506)
            (observe x98 mix 1.9440559976717937)
            (observe x99 mix -0.8963463615041909)
            (observe x100 mix -2.2586454084999916)
            (observe x101 mix 0.16610090546115652)
            (observe x102 mix 1.2912566685236273)
            (observe x103 mix -2.348031682029362)
            (observe x104 mix 0.7416229120251414)
            (observe x105 mix -1.5214314915925307)
            (observe x106 mix -2.552211684596834)
            (observe x107 mix 1.167648388314288)
            (observe x108 mix -1.4399479555621069)
            (observe x109 mix 1.1047155334297223)
            (observe x110 mix 1.3656985711236085)
            (observe x111 mix -1.339195669215786)
            (observe x112 mix -0.5842463805768738)
            (observe x113 mix -0.4406960784148254)
            (observe x114 mix 0.9647790470472418)
            (observe x115 mix -0.842507868715346)
            (observe x116 mix 1.325548494134146)
            (observe x117 mix 0.051665115815942464)
            (observe x118 mix -1.6364231723895002)
            (observe x119 mix -0.7676850033173305)
            (observe x120 mix 2.068367571916448)
            (observe x121 mix -0.7009227275271864)
            (observe x122 mix 0.07666307096391711)
            (observe x123 mix -2.078270365301739)
            (observe x124 mix 0.14085233322660973)
            (observe x125 mix 0.27769984671268544)
            (observe x126 mix -2.1240124524458555)
            (observe x127 mix 0.6676919252599613)
            (observe x128 mix -2.116988243019706)
            (observe x129 mix -0.48271391025697297)
            (observe x130 mix -1.470565931756242)
            (observe x131 mix -0.8404135864207861)
            (observe x132 mix -1.2736703230331161)
            (observe x133 mix 1.9137739197560966)
            (observe x134 mix -1.794330185861237)
            (observe x135 mix 0.6738278282395522)
            (observe x136 mix 1.7745365621992983)
            (observe x137 mix 0.9829605186210754)
            (observe x138 mix -1.1599031883861701)
            (observe x139 mix 1.9946036124638868)
            (observe x140 mix -1.1460861013525065)
            (observe x141 mix 1.8700869312436397)
            (observe x142 mix -1.0032468459155899)
            (observe x143 mix -1.8209852150270205)
            (observe x144 mix -0.9080515521986717)
            (observe x145 mix 1.5691806145080318)
            (observe x146 mix -0.8473743871983541)
            (observe x147 mix -0.8887402730884444)
            (observe x148 mix -1.8518280860816927)
            (observe x149 mix -1.8668943785939789)
            (observe x150 mix 0.09308918676304123)
            (observe x151 mix 1.0868460917540654)
            (observe x152 mix -2.2704872974179464)
            (observe x153 mix -1.01129705142003)
            (observe x154 mix 1.0674244296643216)
            (observe x155 mix 0.97021853308425)
            (observe x156 mix 0.027109017814951675)
            (observe x157 mix 3.2413490207443614)
            (observe x158 mix 0.18048862784020192)
            (observe x159 mix 1.9307860532993346)
            (observe x160 mix 0.8123491031260217)
            (observe x161 mix 0.3031957253853501)
            (observe x162 mix -1.1334002940365062)
            (observe x163 mix 1.0054729453223779)
            (observe x164 mix -1.123228341737473)
            (observe x165 mix 0.34086128836077745)
            (observe x166 mix -1.9306951832528991)
            (observe x167 mix -0.31186327669477254)
            (observe x168 mix -2.1694883720807008)
            (observe x169 mix -0.36669368836972116)
            (observe x170 mix 1.206647737563046)
            (observe x171 mix 1.617355662000283)
            (observe x172 mix 0.32515160869378557)
            (observe x173 mix -0.8787743171521284)
            (observe x174 mix 0.07580849373211995)
            (observe x175 mix 1.0479120714115335)
            (observe x176 mix -0.677113363129402)
            (observe x177 mix 0.6826324171730207)
            (observe x178 mix 2.3537813612740575)
            (observe x179 mix -1.4597249860633905)
            (observe x180 mix 0.27511365330492243)
            (observe x181 mix -1.2915228922584283)
            (observe x182 mix 1.6767325019484072)
            (observe x183 mix 0.8735643425793781)
            (observe x184 mix 1.1194721032651809)
            (observe x185 mix 0.3763854796308129)
            (observe x186 mix 2.041766745744373)
            (observe x187 mix 0.9651872884013396)
            (observe x188 mix -1.7668283008246752)
            (observe x189 mix 0.9600579084473307)
            (observe x190 mix 1.2041630165049422)
            (observe x191 mix -2.294237000903996)
            (observe x192 mix 0.7716612512065488)
            (observe x193 mix 0.7229995753406493)
            (observe x194 mix -2.465491586547344)
            (observe x195 mix -0.49484652270476337)
            (observe x196 mix -0.19965064350454953)
            (observe x197 mix 0.7867700961413551)
            (observe x198 mix -1.0557808035505016)
            (observe x199 mix -1.678998377846264)
        );

        let mut scales = HashMap::new();
        scales.insert("mu1".to_string(), 1.0);
        scales.insert("mu2".to_string(), 1.0);

        let program = GenerativeFunction::new(model, vec![], scales, 42);

        let mut trace = program.simulate(vec![]).unwrap();

        // Print initial mu
        if let Value::Float(mu1) = trace.get_choice(&"mu1".to_string()).value {
            println!("Initial mu1: {}", mu1);
        }
        if let Value::Float(mu2) = trace.get_choice(&"mu2".to_string()).value {
            println!("Initial mu1: {}", mu2);
        }

        let selection = HashSet::from_iter(vec!["mu1".to_string(), "mu2".to_string()]);
        for i in 0..n {
            let (new_trace, weight) = mh(program.clone(), trace, selection.clone()).unwrap();
            if i % 10 == 0 {
                if let Value::Float(mu1) = new_trace.get_choice(&"mu1".to_string()).value {
                    println!("Warmup step {}: mu1 = {}, weight = {}", i, mu1, weight);
                }
                if let Value::Float(mu2) = new_trace.get_choice(&"mu2".to_string()).value {
                    println!("Warmup step {}: mu2 = {}, weight = {}", i, mu2, weight);
                }
            }
            trace = new_trace;
        }

        let mut samples_mu1 = Vec::new();
        let mut samples_mu2 = Vec::new();
        for i in 0..n {
            let (new_trace, accepted) = mh(program.clone(), trace, selection.clone()).unwrap();
            if i % 10 == 0 {
                if let Value::Float(mu1) = new_trace.get_choice(&"mu1".to_string()).value {
                    println!("Sample step {}: mu = {}, accepted = {}", i, mu1, accepted);
                }
                if let Value::Float(mu2) = new_trace.get_choice(&"mu2".to_string()).value {
                    println!("Sample step {}: mu = {}, accepted = {}", i, mu2, accepted);
                }
            }
            if accepted {
                if let Value::Float(mu1) = new_trace.get_choice(&"mu1".to_string()).value {
                    samples_mu1.push(mu1);
                }
                if let Value::Float(mu2) = new_trace.get_choice(&"mu2".to_string()).value {
                    samples_mu2.push(mu2);
                }
            }
            trace = new_trace;
        }

        let mean_mu1: f64 = samples_mu1.iter().sum::<f64>() / samples_mu1.len() as f64;

        let variance_mu1: f64 = samples_mu1
            .iter()
            .map(|x| (x - mean_mu1).powi(2))
            .sum::<f64>()
            / samples_mu1.len() as f64;

        println!(
            "(Gaussian) mu1 mean: {:.3}, var: {:.3}",
            mean_mu1, variance_mu1
        );

        assert!((mean_mu1 + 1.0).abs() < 0.5);
        assert!(variance_mu1 > 0.0 && variance_mu1 < 2.0);

        let mean_mu2: f64 = samples_mu2.iter().sum::<f64>() / samples_mu2.len() as f64;
        let variance_mu2: f64 = samples_mu2
            .iter()
            .map(|x| (x - mean_mu2).powi(2))
            .sum::<f64>()
            / samples_mu2.len() as f64;

        println!(
            "(Gaussian) mu2 mean: {:.3}, var: {:.3}",
            mean_mu2, variance_mu2
        );

        assert!((mean_mu2 - 1.0).abs() < 0.5);
        assert!(variance_mu2 > 0.0 && variance_mu2 < 2.0);
    }

    #[test]
    fn test_model_compact() {
        let mut rng = rand::thread_rng();

        let n = 1000;
        let num_samples = 200;
        let mu1 = 1.0;
        let mu2 = -1.0;
        let sigma = 1.0;
        let p = 0.5;
        let z_dist = Bernoulli::new(p).unwrap();
        let z: Vec<bool> = (0..num_samples).map(|_| z_dist.sample(&mut rng)).collect();

        let component1 = Normal::new(mu1, sigma).unwrap();
        let c1: Vec<f64> = (0..num_samples)
            .map(|_| component1.sample(&mut rng))
            .collect();

        let component2 = Normal::new(mu2, sigma).unwrap();
        let c2: Vec<f64> = (0..num_samples)
            .map(|_| component2.sample(&mut rng))
            .collect();

        let data: Vec<f64> = (0..num_samples)
            .map(|i| if z[i] { c1[i] } else { c2[i] })
            .collect();
        let wrapped_data = Value::List(data.into_iter().map(|x| Value::Float(x)).collect());

        let model = gen!(
            // Priors
            (sample mu1 (normal 0.0 1.0))
            (sample mu2 (normal 0.0 1.0))

            // Ordering
            (constrain (< mu1 mu2))

            // Mixture
            (define p 0.5)
            (define mix (mixture (list (normal mu1 1.0) (normal mu2 1.0)) (list p (- 1.0 p))))

            (define observe-point (lambda (x) (observe (gensym) mix x)))

            (for-each observe-point data)
        );

        let mut scales = HashMap::new();
        scales.insert("mu1".to_string(), 1.0);
        scales.insert("mu2".to_string(), 1.0);

        let program = GenerativeFunction::new(model, vec!["data".to_string()], scales, 42);

        let mut trace = program.simulate(vec![wrapped_data]).unwrap();

        // Print initial mu
        if let Value::Float(mu1) = trace.get_choice(&"mu1".to_string()).value {
            println!("Initial mu1: {}", mu1);
        }
        if let Value::Float(mu2) = trace.get_choice(&"mu2".to_string()).value {
            println!("Initial mu1: {}", mu2);
        }

        let selection = HashSet::from_iter(vec!["mu1".to_string(), "mu2".to_string()]);
        for i in 0..n {
            let (new_trace, weight) = mh(program.clone(), trace, selection.clone()).unwrap();
            if i % 10 == 0 {
                if let Value::Float(mu1) = new_trace.get_choice(&"mu1".to_string()).value {
                    println!("Warmup step {}: mu1 = {}, weight = {}", i, mu1, weight);
                }
                if let Value::Float(mu2) = new_trace.get_choice(&"mu2".to_string()).value {
                    println!("Warmup step {}: mu2 = {}, weight = {}", i, mu2, weight);
                }
            }
            trace = new_trace;
        }

        let mut samples_mu1 = Vec::new();
        let mut samples_mu2 = Vec::new();
        for i in 0..n {
            let (new_trace, accepted) = mh(program.clone(), trace, selection.clone()).unwrap();
            if i % 10 == 0 {
                if let Value::Float(mu1) = new_trace.get_choice(&"mu1".to_string()).value {
                    println!("Sample step {}: mu = {}, accepted = {}", i, mu1, accepted);
                }
                if let Value::Float(mu2) = new_trace.get_choice(&"mu2".to_string()).value {
                    println!("Sample step {}: mu = {}, accepted = {}", i, mu2, accepted);
                }
            }
            if accepted {
                if let Value::Float(mu1) = new_trace.get_choice(&"mu1".to_string()).value {
                    samples_mu1.push(mu1);
                }
                if let Value::Float(mu2) = new_trace.get_choice(&"mu2".to_string()).value {
                    samples_mu2.push(mu2);
                }
            }
            trace = new_trace;
        }

        let mean_mu1: f64 = samples_mu1.iter().sum::<f64>() / samples_mu1.len() as f64;

        let variance_mu1: f64 = samples_mu1
            .iter()
            .map(|x| (x - mean_mu1).powi(2))
            .sum::<f64>()
            / samples_mu1.len() as f64;

        println!(
            "(Gaussian) mu1 mean: {:.3}, var: {:.3}",
            mean_mu1, variance_mu1
        );

        let mean_mu2: f64 = samples_mu2.iter().sum::<f64>() / samples_mu2.len() as f64;
        let variance_mu2: f64 = samples_mu2
            .iter()
            .map(|x| (x - mean_mu2).powi(2))
            .sum::<f64>()
            / samples_mu2.len() as f64;

        println!(
            "(Gaussian) mu2 mean: {:.3}, var: {:.3}",
            mean_mu2, variance_mu2
        );

        assert!((mean_mu1 + 1.0).abs() < 0.5);
        assert!(variance_mu1 > 0.0 && variance_mu1 < 2.0);

        assert!((mean_mu2 - 1.0).abs() < 0.5);
        assert!(variance_mu2 > 0.0 && variance_mu2 < 2.0);
    }
}
