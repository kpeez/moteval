//! Linear sum assignment, ported line by line from scipy 1.18.0.
//!
//! Sources: `scipy/optimize/_lsapmodule.c` (the Python entry point: `float64` C-order
//! input, result arrays of length `min(nr, nc)`, error text) and
//! `scipy/optimize/rectangular_lsap/rectangular_lsap.cpp` (the shortest augmenting path
//! algorithm of DF Crouse, "On implementing 2D rectangular assignment algorithms", IEEE
//! TAES 52(4), 2016). Tie-breaks decide which pairs match, so every comparison, every
//! loop order, and every float operation follows the C++ exactly.

use std::fmt;

/// The two errors scipy raises (both `ValueError`), with scipy's message text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LsapError {
    /// A NaN or `-inf` entry (`RECTANGULAR_LSAP_INVALID`).
    InvalidEntries,
    /// No finite assignment exists (`RECTANGULAR_LSAP_INFEASIBLE`).
    Infeasible,
}

impl fmt::Display for LsapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEntries => f.write_str("matrix contains invalid numeric entries"),
            Self::Infeasible => f.write_str("cost matrix is infeasible"),
        }
    }
}

impl std::error::Error for LsapError {}

/// `scipy.optimize.linear_sum_assignment(cost, maximize)` for a C-order `nr x nc` matrix.
///
/// Returns `(row_ind, col_ind)`, each of length `min(nr, nc)`. Without a transpose
/// (`nc >= nr`) the rows are `0..nr` in order; for a tall matrix the pairs are sorted by
/// row.
pub fn linear_sum_assignment(
    cost: &[f64],
    nr: usize,
    nc: usize,
    maximize: bool,
) -> Result<(Vec<usize>, Vec<usize>), LsapError> {
    assert_eq!(cost.len(), nr * nc, "cost matrix data is not nr x nc");

    // handle trivial inputs
    if nr == 0 || nc == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // tall rectangular cost matrix must be transposed
    let transpose = nc < nr;
    let (nr, nc) = if transpose { (nc, nr) } else { (nr, nc) };

    // copy (and transpose, and negate for maximization) only when the input must change
    let owned: Vec<f64>;
    let cost: &[f64] = if transpose || maximize {
        let mut temp = vec![0.0; nr * nc];
        if transpose {
            // the input is (nc x nr) here: element (i, j) of the input goes to (j, i)
            for i in 0..nc {
                for j in 0..nr {
                    temp[j * nc + i] = cost[i * nr + j];
                }
            }
        } else {
            temp.copy_from_slice(cost);
        }
        if maximize {
            for x in &mut temp {
                *x = -*x;
            }
        }
        owned = temp;
        &owned
    } else {
        cost
    };

    // test for NaN and -inf entries
    if cost.iter().any(|&c| c.is_nan() || c == f64::NEG_INFINITY) {
        return Err(LsapError::InvalidEntries);
    }

    // initialize variables
    let mut u = vec![0.0; nr];
    let mut v = vec![0.0; nc];
    let mut shortest_path_costs = vec![0.0; nc];
    let mut path: Vec<Option<usize>> = vec![None; nc];
    let mut col4row: Vec<Option<usize>> = vec![None; nr];
    let mut row4col: Vec<Option<usize>> = vec![None; nc];
    let mut sr = vec![false; nr];
    let mut sc = vec![false; nc];
    let mut remaining = vec![0usize; nc];

    // iteratively build the solution
    for cur_row in 0..nr {
        let (sink, min_val) = augmenting_path(
            nc,
            cost,
            &u,
            &v,
            &mut path,
            &row4col,
            &mut shortest_path_costs,
            cur_row,
            &mut sr,
            &mut sc,
            &mut remaining,
        )
        .ok_or(LsapError::Infeasible)?;

        // update dual variables
        u[cur_row] += min_val;
        for i in 0..nr {
            if sr[i] && i != cur_row {
                let j = col4row[i].expect("a scanned row other than cur_row is assigned");
                u[i] += min_val - shortest_path_costs[j];
            }
        }
        for j in 0..nc {
            if sc[j] {
                v[j] -= min_val - shortest_path_costs[j];
            }
        }

        // augment previous solution
        let mut j = sink;
        loop {
            let i = path[j].expect("every column on the augmenting path has a parent");
            row4col[j] = Some(i);
            let previous = col4row[i].replace(j);
            if i == cur_row {
                break;
            }
            j = previous.expect("a row on the path before cur_row is assigned");
        }
    }

    let col4row: Vec<usize> = col4row
        .into_iter()
        .map(|c| c.expect("every row is assigned"))
        .collect();
    if transpose {
        // argsort_iter(col4row): the values are distinct column indices, so the order of
        // std::sort (unstable) is unique and any sort gives it.
        let mut order: Vec<usize> = (0..nr).collect();
        order.sort_unstable_by_key(|&k| col4row[k]);
        let a = order.iter().map(|&k| col4row[k]).collect();
        Ok((a, order))
    } else {
        Ok(((0..nr).collect(), col4row))
    }
}

/// `augmenting_path` from `rectangular_lsap.cpp`: the sink column and the path length, or
/// `None` for an infeasible matrix.
#[allow(clippy::too_many_arguments)]
fn augmenting_path(
    nc: usize,
    cost: &[f64],
    u: &[f64],
    v: &[f64],
    path: &mut [Option<usize>],
    row4col: &[Option<usize>],
    shortest_path_costs: &mut [f64],
    mut i: usize,
    sr: &mut [bool],
    sc: &mut [bool],
    remaining: &mut [usize],
) -> Option<(usize, f64)> {
    let mut min_val = 0.0;

    // Crouse's pseudocode uses set complements to keep track of remaining nodes; scipy
    // uses a vector. Filling it in reverse makes a constant cost matrix solve to the
    // identity (scipy #11602).
    let mut num_remaining = nc;
    for (it, slot) in remaining.iter_mut().enumerate() {
        *slot = nc - it - 1;
    }

    sr.fill(false);
    sc.fill(false);
    shortest_path_costs.fill(f64::INFINITY);

    // find shortest augmenting path
    let mut sink = None;
    while sink.is_none() {
        let mut index = None;
        let mut lowest = f64::INFINITY;
        sr[i] = true;

        for (it, &j) in remaining[..num_remaining].iter().enumerate() {
            let r = min_val + cost[i * nc + j] - u[i] - v[j];
            if r < shortest_path_costs[j] {
                path[j] = Some(i);
                shortest_path_costs[j] = r;
            }

            // When several nodes share the minimum cost, prefer one that gives a new sink
            // node. This matters for integer cost matrices with small coefficients.
            if shortest_path_costs[j] < lowest
                || (shortest_path_costs[j] == lowest && row4col[j].is_none())
            {
                lowest = shortest_path_costs[j];
                index = Some(it);
            }
        }

        min_val = lowest;
        if min_val == f64::INFINITY {
            // infeasible cost matrix
            return None;
        }

        let index = index.expect("a finite lowest cost has an index");
        let j = remaining[index];
        match row4col[j] {
            None => sink = Some(j),
            Some(row) => i = row,
        }

        sc[j] = true;
        num_remaining -= 1;
        remaining[index] = remaining[num_remaining];
    }

    sink.map(|s| (s, min_val))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_and_error_inputs() {
        assert_eq!(
            linear_sum_assignment(&[], 0, 3, false),
            Ok((vec![], vec![]))
        );
        assert_eq!(
            linear_sum_assignment(&[1.0, f64::NAN], 1, 2, false),
            Err(LsapError::InvalidEntries)
        );
        assert_eq!(
            linear_sum_assignment(&[f64::NEG_INFINITY], 1, 1, false),
            Err(LsapError::InvalidEntries)
        );
        assert_eq!(
            linear_sum_assignment(&[f64::INFINITY, f64::INFINITY], 1, 2, false),
            Err(LsapError::Infeasible)
        );
        // maximize negates first, so +inf becomes the rejected -inf
        assert_eq!(
            linear_sum_assignment(&[f64::INFINITY], 1, 1, true),
            Err(LsapError::InvalidEntries)
        );
    }

    #[test]
    fn constant_matrix_solves_to_identity() {
        let (rows, cols) = linear_sum_assignment(&[1.0; 9], 3, 3, false).unwrap();
        assert_eq!(rows, vec![0, 1, 2]);
        assert_eq!(cols, vec![0, 1, 2]);
    }

    #[test]
    fn tall_matrix_is_transposed_and_sorted_by_row() {
        // 3 x 2: rows 2 and 0 are cheapest for columns 0 and 1
        let cost = [5.0, 1.0, 9.0, 9.0, 0.0, 9.0];
        let (rows, cols) = linear_sum_assignment(&cost, 3, 2, false).unwrap();
        assert_eq!(rows, vec![0, 2]);
        assert_eq!(cols, vec![1, 0]);
    }
}
