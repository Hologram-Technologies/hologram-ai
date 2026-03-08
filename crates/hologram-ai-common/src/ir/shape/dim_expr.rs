use std::collections::HashSet;

/// Compact interned identifier for a dimension variable.
/// Points into the `DimVarTable` for name and bounds resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct DimVarId(pub(crate) u32);

/// A dimension expression supporting the algebra needed for ML shape inference.
///
/// Deliberately limited to the operations needed for ML shape rules:
/// arithmetic for Reshape product constraints, CeilDiv for padding/tiling,
/// Max for broadcast, Min for clamp/bound. Not a general CAS.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimExpr {
    /// A known constant dimension.
    Concrete(u64),
    /// A symbolic variable (batch_size, seq_len, etc.).
    Var(DimVarId),
    /// Arithmetic operations.
    Add(Box<DimExpr>, Box<DimExpr>),
    Sub(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    Div(Box<DimExpr>, Box<DimExpr>),
    Mod(Box<DimExpr>, Box<DimExpr>),
    /// Ceiling division: ceil(a / b) = (a + b - 1) / b.
    CeilDiv(Box<DimExpr>, Box<DimExpr>),
    /// Maximum of two expressions. Used in broadcast rules.
    Max(Box<DimExpr>, Box<DimExpr>),
    /// Minimum of two expressions. Used in clamp/bound calculations.
    Min(Box<DimExpr>, Box<DimExpr>),
    /// Truly unknown dimension — cannot be expressed symbolically.
    Dynamic,
}

impl DimExpr {
    pub fn concrete(n: u64) -> Self {
        DimExpr::Concrete(n)
    }

    pub fn var(id: DimVarId) -> Self {
        DimExpr::Var(id)
    }

    /// Returns the concrete value if this is a `Concrete` variant.
    pub fn as_concrete(&self) -> Option<u64> {
        match self {
            DimExpr::Concrete(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns true if this expression contains no `Var` or `Dynamic` nodes.
    pub fn is_concrete(&self) -> bool {
        self.evaluate().is_some()
    }

    /// Attempt to evaluate to a concrete u64.
    /// Returns `None` if any `Var` or `Dynamic` is encountered.
    /// Division by zero returns `None`.
    pub fn evaluate(&self) -> Option<u64> {
        match self {
            DimExpr::Concrete(v) => Some(*v),
            DimExpr::Var(_) | DimExpr::Dynamic => None,
            DimExpr::Add(a, b) => Some(a.evaluate()?.checked_add(b.evaluate()?)?),
            DimExpr::Sub(a, b) => Some(a.evaluate()?.checked_sub(b.evaluate()?)?),
            DimExpr::Mul(a, b) => Some(a.evaluate()?.checked_mul(b.evaluate()?)?),
            DimExpr::Div(a, b) => {
                let bv = b.evaluate()?;
                if bv == 0 { return None; }
                Some(a.evaluate()? / bv)
            }
            DimExpr::Mod(a, b) => {
                let bv = b.evaluate()?;
                if bv == 0 { return None; }
                Some(a.evaluate()? % bv)
            }
            DimExpr::CeilDiv(a, b) => {
                let av = a.evaluate()?;
                let bv = b.evaluate()?;
                if bv == 0 { return None; }
                Some(av.div_ceil(bv))
            }
            DimExpr::Max(a, b) => Some(a.evaluate()?.max(b.evaluate()?)),
            DimExpr::Min(a, b) => Some(a.evaluate()?.min(b.evaluate()?)),
        }
    }

    /// Substitute all occurrences of `var` with `value`.
    pub fn substitute(&self, var: DimVarId, value: &DimExpr) -> DimExpr {
        match self {
            DimExpr::Concrete(_) | DimExpr::Dynamic => self.clone(),
            DimExpr::Var(id) if *id == var => value.clone(),
            DimExpr::Var(_) => self.clone(),
            DimExpr::Add(a, b) => DimExpr::Add(
                Box::new(a.substitute(var, value)),
                Box::new(b.substitute(var, value)),
            ),
            DimExpr::Sub(a, b) => DimExpr::Sub(
                Box::new(a.substitute(var, value)),
                Box::new(b.substitute(var, value)),
            ),
            DimExpr::Mul(a, b) => DimExpr::Mul(
                Box::new(a.substitute(var, value)),
                Box::new(b.substitute(var, value)),
            ),
            DimExpr::Div(a, b) => DimExpr::Div(
                Box::new(a.substitute(var, value)),
                Box::new(b.substitute(var, value)),
            ),
            DimExpr::Mod(a, b) => DimExpr::Mod(
                Box::new(a.substitute(var, value)),
                Box::new(b.substitute(var, value)),
            ),
            DimExpr::CeilDiv(a, b) => DimExpr::CeilDiv(
                Box::new(a.substitute(var, value)),
                Box::new(b.substitute(var, value)),
            ),
            DimExpr::Max(a, b) => DimExpr::Max(
                Box::new(a.substitute(var, value)),
                Box::new(b.substitute(var, value)),
            ),
            DimExpr::Min(a, b) => DimExpr::Min(
                Box::new(a.substitute(var, value)),
                Box::new(b.substitute(var, value)),
            ),
        }
    }

    /// Simplify constant sub-expressions.
    /// E.g., `Add(Concrete(3), Concrete(5))` => `Concrete(8)`.
    pub fn simplify(&self) -> DimExpr {
        if let Some(v) = self.evaluate() {
            return DimExpr::Concrete(v);
        }
        match self {
            DimExpr::Concrete(_) | DimExpr::Var(_) | DimExpr::Dynamic => self.clone(),
            DimExpr::Add(a, b) => DimExpr::Add(Box::new(a.simplify()), Box::new(b.simplify())),
            DimExpr::Sub(a, b) => DimExpr::Sub(Box::new(a.simplify()), Box::new(b.simplify())),
            DimExpr::Mul(a, b) => DimExpr::Mul(Box::new(a.simplify()), Box::new(b.simplify())),
            DimExpr::Div(a, b) => DimExpr::Div(Box::new(a.simplify()), Box::new(b.simplify())),
            DimExpr::Mod(a, b) => DimExpr::Mod(Box::new(a.simplify()), Box::new(b.simplify())),
            DimExpr::CeilDiv(a, b) => DimExpr::CeilDiv(Box::new(a.simplify()), Box::new(b.simplify())),
            DimExpr::Max(a, b) => DimExpr::Max(Box::new(a.simplify()), Box::new(b.simplify())),
            DimExpr::Min(a, b) => DimExpr::Min(Box::new(a.simplify()), Box::new(b.simplify())),
        }
    }

    /// Collect all `DimVarId`s referenced in this expression.
    pub fn free_vars(&self) -> HashSet<DimVarId> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, out: &mut HashSet<DimVarId>) {
        match self {
            DimExpr::Concrete(_) | DimExpr::Dynamic => {}
            DimExpr::Var(id) => { out.insert(*id); }
            DimExpr::Add(a, b) | DimExpr::Sub(a, b) | DimExpr::Mul(a, b)
            | DimExpr::Div(a, b) | DimExpr::Mod(a, b) | DimExpr::CeilDiv(a, b)
            | DimExpr::Max(a, b) | DimExpr::Min(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
        }
    }
}

impl From<u64> for DimExpr {
    fn from(v: u64) -> Self { DimExpr::Concrete(v) }
}

impl From<usize> for DimExpr {
    fn from(v: usize) -> Self { DimExpr::Concrete(v as u64) }
}

impl serde::Serialize for DimExpr {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            DimExpr::Concrete(v) => serializer.serialize_u64(*v),
            DimExpr::Dynamic => serializer.serialize_str("?"),
            DimExpr::Var(id) => serializer.serialize_str(&format!("var:{}", id.0)),
            _ => serializer.serialize_str(&format!("{self:?}")),
        }
    }
}

impl<'de> serde::Deserialize<'de> for DimExpr {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de;
        struct DimExprVisitor;
        impl<'de> de::Visitor<'de> for DimExprVisitor {
            type Value = DimExpr;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a u64 or string")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<DimExpr, E> {
                Ok(DimExpr::Concrete(v))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<DimExpr, E> {
                if v == "?" {
                    Ok(DimExpr::Dynamic)
                } else if let Some(rest) = v.strip_prefix("var:") {
                    let id: u32 = rest.parse().map_err(de::Error::custom)?;
                    Ok(DimExpr::Var(DimVarId(id)))
                } else {
                    Err(de::Error::custom(format!("unrecognized DimExpr: {v}")))
                }
            }
        }
        deserializer.deserialize_any(DimExprVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concrete_evaluate() {
        let e = DimExpr::Add(
            Box::new(DimExpr::Concrete(3)),
            Box::new(DimExpr::Concrete(5)),
        );
        assert_eq!(e.evaluate(), Some(8));
        assert!(e.is_concrete());
    }

    #[test]
    fn var_not_evaluable() {
        let e = DimExpr::Var(DimVarId(0));
        assert_eq!(e.evaluate(), None);
        assert!(!e.is_concrete());
    }

    #[test]
    fn substitute_var() {
        let v = DimVarId(0);
        let e = DimExpr::Add(
            Box::new(DimExpr::Var(v)),
            Box::new(DimExpr::Concrete(1)),
        );
        let subst = e.substitute(v, &DimExpr::Concrete(10));
        assert_eq!(subst.evaluate(), Some(11));
    }

    #[test]
    fn simplify_partial() {
        let e = DimExpr::Add(
            Box::new(DimExpr::Add(
                Box::new(DimExpr::Concrete(2)),
                Box::new(DimExpr::Concrete(3)),
            )),
            Box::new(DimExpr::Var(DimVarId(0))),
        );
        let s = e.simplify();
        // Inner Add(2,3) should simplify to Concrete(5), outer still has Var
        match &s {
            DimExpr::Add(a, _) => assert_eq!(a.as_concrete(), Some(5)),
            _ => panic!("expected Add"),
        }
    }

    #[test]
    fn free_vars_collection() {
        let v0 = DimVarId(0);
        let v1 = DimVarId(1);
        let e = DimExpr::Mul(
            Box::new(DimExpr::Var(v0)),
            Box::new(DimExpr::Add(
                Box::new(DimExpr::Var(v1)),
                Box::new(DimExpr::Concrete(1)),
            )),
        );
        let vars = e.free_vars();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&v0));
        assert!(vars.contains(&v1));
    }

    #[test]
    fn ceil_div_evaluate() {
        let e = DimExpr::CeilDiv(
            Box::new(DimExpr::Concrete(7)),
            Box::new(DimExpr::Concrete(3)),
        );
        assert_eq!(e.evaluate(), Some(3)); // ceil(7/3) = 3
    }

    #[test]
    fn div_by_zero_returns_none() {
        let e = DimExpr::Div(
            Box::new(DimExpr::Concrete(10)),
            Box::new(DimExpr::Concrete(0)),
        );
        assert_eq!(e.evaluate(), None);
    }

    #[test]
    fn as_concrete_compat() {
        assert_eq!(DimExpr::Concrete(42).as_concrete(), Some(42));
        assert_eq!(DimExpr::Dynamic.as_concrete(), None);
        assert_eq!(DimExpr::Var(DimVarId(0)).as_concrete(), None);
    }

    #[test]
    fn from_impls() {
        let d: DimExpr = 42u64.into();
        assert_eq!(d.as_concrete(), Some(42));
        let d: DimExpr = 7usize.into();
        assert_eq!(d.as_concrete(), Some(7));
    }
}
