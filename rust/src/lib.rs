//! Rust accelerated LP writer for nimopt

use numpy::{PyReadonlyArray1, PyArray1, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

#[pyfunction]
fn write_lp_header(filename: &str, model_name: &str, sense: &str) -> PyResult<()> {
    let mut file = File::create(filename)?;
    writeln!(file, "\\ {}", model_name)?;
    writeln!(file, "\\ nimopt v2 (rust)\n")?;
    writeln!(file, "{}\n", if sense == "minimize" { "Minimize" } else { "Maximize" })?;
    Ok(())
}

#[pyfunction]
fn write_objective(
    filename: &str,
    var_names: Vec<String>,
    coefficients: PyReadonlyArray1<'_, f64>,
) -> PyResult<()> {
    let coefs = coefficients.as_slice()?;
    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::new(file);
    write!(w, " obj:")?;
    let mut first = true;
    for (i, name) in var_names.iter().enumerate() {
        let c = coefs.get(i).copied().unwrap_or(1.0);
        if c == 0.0 { continue; }
        write_term(&mut w, c, name, first)?;
        first = false;
    }
    if first { write!(w, " 0")?; }
    writeln!(w, "\n")?;
    Ok(())
}

/// Write objective with variable names generated in Rust
#[pyfunction]
fn write_objective_fast(
    filename: &str,
    var_name: &str,
    dim_elements: Vec<Vec<String>>,
    coefficients: PyReadonlyArray1<'_, f64>,
) -> PyResult<()> {
    let coefs = coefficients.as_slice()?;
    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::new(file);
    write!(w, " obj:")?;
    
    let combos = cartesian_product_ref(&dim_elements);
    let mut first = true;
    
    for (i, combo) in combos.iter().enumerate() {
        let c = coefs.get(i).copied().unwrap_or(1.0);
        if c == 0.0 { continue; }
        
        // Build variable name
        let mut vname = var_name.to_string();
        for (d, &idx) in combo.iter().enumerate() {
            vname.push('_');
            vname.push_str(&dim_elements[d][idx]);
        }
        
        write_term(&mut w, c, &vname, first)?;
        first = false;
    }
    
    if first { write!(w, " 0")?; }
    writeln!(w, "\n")?;
    Ok(())
}

#[pyfunction]
fn write_subject_to(filename: &str) -> PyResult<()> {
    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "Subject To")?;
    Ok(())
}

#[pyfunction]
fn write_uniform_constraints(
    py: Python<'_>,
    filename: &str,
    name_prefix: &str,
    sense: &str,
    var_names: Vec<String>,
    rhs: PyReadonlyArray1<'_, f64>,
    vars_per_con: usize,
    coef: f64,
) -> PyResult<()> {
    let rhs_vals = rhs.as_slice()?;
    let n_cons = var_names.len() / vars_per_con;
    let sense_str = match sense { "<=" => "<=", ">=" => ">=", _ => "=" };
    let lines: Vec<String> = py.allow_threads(|| {
        (0..n_cons).into_par_iter().map(|i| {
            let mut s = format!(" {}_{}: ", name_prefix, i);
            for j in 0..vars_per_con {
                let vname = &var_names[i * vars_per_con + j];
                if j == 0 {
                    if coef == 1.0 { s.push_str(vname); }
                    else if coef == -1.0 { s.push_str("- "); s.push_str(vname); }
                    else { s.push_str(&format!("{} {}", coef, vname)); }
                } else {
                    if coef == 1.0 { s.push_str(" + "); s.push_str(vname); }
                    else if coef == -1.0 { s.push_str(" - "); s.push_str(vname); }
                    else if coef > 0.0 { s.push_str(&format!(" + {} {}", coef, vname)); }
                    else { s.push_str(&format!(" - {} {}", -coef, vname)); }
                }
            }
            let rval = rhs_vals.get(i).copied().unwrap_or(0.0);
            s.push_str(&format!(" {} {}\n", sense_str, rval));
            s
        }).collect()
    });
    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, file);
    for line in lines { w.write_all(line.as_bytes())?; }
    w.flush()?;
    Ok(())
}

#[pyfunction]
fn write_coef_constraints(
    py: Python<'_>,
    filename: &str,
    name_prefix: &str,
    sense: &str,
    var_names: Vec<String>,
    coefficients: PyReadonlyArray1<'_, f64>,
    rhs: PyReadonlyArray1<'_, f64>,
    vars_per_con: usize,
) -> PyResult<()> {
    let coefs = coefficients.as_slice()?;
    let rhs_vals = rhs.as_slice()?;
    let n_cons = var_names.len() / vars_per_con;
    let sense_str = match sense { "<=" => "<=", ">=" => ">=", _ => "=" };
    let lines: Vec<String> = py.allow_threads(|| {
        (0..n_cons).into_par_iter().map(|i| {
            let mut s = format!(" {}_{}: ", name_prefix, i);
            let mut first = true;
            for j in 0..vars_per_con {
                let idx = i * vars_per_con + j;
                let c = coefs.get(idx).copied().unwrap_or(1.0);
                if c == 0.0 { continue; }
                let vname = &var_names[idx];
                append_term(&mut s, c, vname, first);
                first = false;
            }
            if first { s.push_str("0"); }
            let rval = rhs_vals.get(i).copied().unwrap_or(0.0);
            s.push_str(&format!(" {} {}\n", sense_str, rval));
            s
        }).collect()
    });
    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, file);
    for line in lines { w.write_all(line.as_bytes())?; }
    w.flush()?;
    Ok(())
}

/// Generate constraints with variable names built in Rust.
/// 
/// var_dim_elements: list of element lists in VARIABLE dimension order
/// is_free_dim: for each dimension, true if iterating over constraints, false if summing
/// coef_flat: flattened coefficients (if any)
/// coef_shape: shape of coefficient array in variable dimension order
/// rhs_flat: RHS values, one per constraint
#[pyfunction]
#[pyo3(signature = (filename, con_name, sense, var_name, var_dim_elements, is_free_dim, coef_flat, coef_shape, rhs_flat))]
fn write_sum_constraints(
    py: Python<'_>,
    filename: &str,
    con_name: &str,
    sense: &str,
    var_name: &str,
    var_dim_elements: Vec<Vec<String>>,
    is_free_dim: Vec<bool>,
    coef_flat: Option<PyReadonlyArray1<'_, f64>>,
    coef_shape: Vec<usize>,
    rhs_flat: PyReadonlyArray1<'_, f64>,
) -> PyResult<()> {
    let rhs_vals = rhs_flat.as_slice()?;
    let coef_data: Option<Vec<f64>> = coef_flat.map(|c| c.as_slice().unwrap().to_vec());
    let sense_str = match sense { "<=" => "<=", ">=" => ">=", _ => "=" };

    // Compute strides for coefficient indexing in variable dimension order
    let mut coef_strides = vec![1usize; coef_shape.len()];
    for i in (0..coef_shape.len().saturating_sub(1)).rev() {
        coef_strides[i] = coef_strides[i + 1] * coef_shape[i + 1];
    }

    // Compute strides for variable dimensions
    let ndim = var_dim_elements.len();
    let mut var_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        var_strides[i] = var_strides[i + 1] * var_dim_elements[i + 1].len();
    }

    // Separate free and sum dimensions
    let mut free_dims: Vec<usize> = Vec::new();
    let mut sum_dims: Vec<usize> = Vec::new();
    for (i, &is_free) in is_free_dim.iter().enumerate() {
        if is_free {
            free_dims.push(i);
        } else {
            sum_dims.push(i);
        }
    }

    // Generate free index combinations (these determine constraints)
    let free_elements: Vec<Vec<String>> = free_dims.iter().map(|&d| var_dim_elements[d].clone()).collect();
    let free_elements_ref: Vec<&Vec<String>> = free_elements.iter().collect();
    let free_combos = cartesian_product(&free_elements_ref);
    let n_cons = free_combos.len();

    // Generate sum index combinations (these are summed within each constraint)
    let sum_elements: Vec<&Vec<String>> = sum_dims.iter().map(|&d| &var_dim_elements[d]).collect();
    let sum_combos = cartesian_product(&sum_elements);

    let lines: Vec<String> = py.allow_threads(|| {
        free_combos.into_par_iter().enumerate().map(|(con_idx, free_combo)| {
            // Build constraint name using element names
            let mut con_suffix = String::new();
            for (i, &idx) in free_combo.iter().enumerate() {
                con_suffix.push('_');
                con_suffix.push_str(&free_elements[i][idx]);
            }
            let mut s = format!(" {}{}: ", con_name, con_suffix);
            let mut first = true;

            for sum_combo in &sum_combos {
                // Build full index vector in variable dimension order
                let mut full_idx = vec![0usize; ndim];
                let mut fi = 0;
                let mut si = 0;
                for d in 0..ndim {
                    if is_free_dim[d] {
                        full_idx[d] = free_combo[fi];
                        fi += 1;
                    } else {
                        full_idx[d] = sum_combo[si];
                        si += 1;
                    }
                }

                // Build variable name
                let mut vname = var_name.to_string();
                for (d, &idx) in full_idx.iter().enumerate() {
                    vname.push('_');
                    vname.push_str(&var_dim_elements[d][idx]);
                }

                // Get coefficient
                let c = if let Some(ref coef) = coef_data {
                    // Map variable dimensions to coefficient dimensions
                    let mut flat_idx = 0;
                    for (d, &idx) in full_idx.iter().enumerate() {
                        if d < coef_strides.len() {
                            flat_idx += idx * coef_strides[d];
                        }
                    }
                    coef.get(flat_idx).copied().unwrap_or(1.0)
                } else {
                    1.0
                };

                if c != 0.0 {
                    append_term(&mut s, c, &vname, first);
                    first = false;
                }
            }

            if first { s.push_str("0"); }
            let rval = rhs_vals.get(con_idx).copied().unwrap_or(0.0);
            s.push_str(&format!(" {} {}\n", sense_str, rval));
            s
        }).collect()
    });

    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, file);
    for line in lines { w.write_all(line.as_bytes())?; }
    w.flush()?;
    Ok(())
}

fn cartesian_product(elements: &[&Vec<String>]) -> Vec<Vec<usize>> {
    if elements.is_empty() { return vec![vec![]]; }
    let mut result = vec![vec![]];
    for dim in elements {
        let mut new_result = Vec::with_capacity(result.len() * dim.len());
        for combo in &result {
            for idx in 0..dim.len() {
                let mut new_combo = combo.clone();
                new_combo.push(idx);
                new_result.push(new_combo);
            }
        }
        result = new_result;
    }
    result
}

#[inline]
fn append_term(s: &mut String, c: f64, name: &str, first: bool) {
    if first {
        if c == 1.0 { s.push_str(name); }
        else if c == -1.0 { s.push_str("- "); s.push_str(name); }
        else if c > 0.0 { s.push_str(&format!("{} {}", c, name)); }
        else { s.push_str(&format!("- {} {}", -c, name)); }
    } else {
        if c == 1.0 { s.push_str(" + "); s.push_str(name); }
        else if c == -1.0 { s.push_str(" - "); s.push_str(name); }
        else if c > 0.0 { s.push_str(&format!(" + {} {}", c, name)); }
        else { s.push_str(&format!(" - {} {}", -c, name)); }
    }
}

#[pyfunction]
fn write_bounds(
    filename: &str,
    var_names: Vec<String>,
    lbs: Vec<Option<f64>>,
    ubs: Vec<Option<f64>>,
) -> PyResult<()> {
    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "\nBounds")?;
    for (i, name) in var_names.iter().enumerate() {
        let lb = lbs.get(i).copied().flatten();
        let ub = ubs.get(i).copied().flatten();
        match (lb, ub) {
            (None, None) => writeln!(w, " {} free", name)?,
            (None, Some(u)) => writeln!(w, " {} <= {}", name, u)?,
            (Some(l), None) => writeln!(w, " {} >= {}", name, l)?,
            (Some(l), Some(u)) => writeln!(w, " {} <= {} <= {}", l, name, u)?,
        }
    }
    Ok(())
}

#[pyfunction]
fn write_var_types(
    filename: &str,
    integers: Vec<String>,
    binaries: Vec<String>,
) -> PyResult<()> {
    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::new(file);
    if !integers.is_empty() {
        writeln!(w, "\nGeneral")?;
        for n in integers { writeln!(w, " {}", n)?; }
    }
    if !binaries.is_empty() {
        writeln!(w, "\nBinary")?;
        for n in binaries { writeln!(w, " {}", n)?; }
    }
    writeln!(w, "\nEnd")?;
    Ok(())
}

fn write_term<W: Write>(w: &mut W, c: f64, name: &str, first: bool) -> std::io::Result<()> {
    if first {
        if c == 1.0 { write!(w, " {}", name)?; }
        else if c == -1.0 { write!(w, " - {}", name)?; }
        else if c > 0.0 { write!(w, " {} {}", c, name)?; }
        else { write!(w, " - {} {}", -c, name)?; }
    } else {
        if c == 1.0 { write!(w, " + {}", name)?; }
        else if c == -1.0 { write!(w, " - {}", name)?; }
        else if c > 0.0 { write!(w, " + {} {}", c, name)?; }
        else { write!(w, " - {} {}", -c, name)?; }
    }
    Ok(())
}


/// Write a single constraint with many variables (no free sets)
#[pyfunction]
fn write_single_constraint(
    filename: &str,
    con_name: &str,
    sense: &str,
    var_names: Vec<String>,
    coefficients: PyReadonlyArray1<'_, f64>,
    rhs: f64,
) -> PyResult<()> {
    let coefs = coefficients.as_slice()?;
    let sense_str = match sense { "<=" => "<=", ">=" => ">=", _ => "=" };

    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::new(file);

    write!(w, " {}: ", con_name)?;
    let mut first = true;
    for (i, name) in var_names.iter().enumerate() {
        let c = coefs.get(i).copied().unwrap_or(1.0);
        if c == 0.0 { continue; }
        if first {
            if c == 1.0 { write!(w, "{}", name)?; }
            else if c == -1.0 { write!(w, "- {}", name)?; }
            else if c > 0.0 { write!(w, "{} {}", c, name)?; }
            else { write!(w, "- {} {}", -c, name)?; }
            first = false;
        } else {
            if c == 1.0 { write!(w, " + {}", name)?; }
            else if c == -1.0 { write!(w, " - {}", name)?; }
            else if c > 0.0 { write!(w, " + {} {}", c, name)?; }
            else { write!(w, " - {} {}", -c, name)?; }
        }
    }
    if first { write!(w, "0")?; }
    writeln!(w, " {} {}", sense_str, rhs)?;
    Ok(())
}

/// Write bounds with variable names generated in Rust
#[pyfunction]
fn write_bounds_fast(
    filename: &str,
    var_name: &str,
    dim_elements: Vec<Vec<String>>,
    lb: Option<f64>,
    ub: Option<f64>,
) -> PyResult<()> {
    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::with_capacity(4 * 1024 * 1024, file);

    // Generate all variable names and write bounds
    let combos = cartesian_product_ref(&dim_elements);
    
    for combo in combos {
        let mut vname = var_name.to_string();
        for (d, idx) in combo.iter().enumerate() {
            vname.push('_');
            vname.push_str(&dim_elements[d][*idx]);
        }
        
        match (lb, ub) {
            (None, None) => writeln!(w, " {} free", vname)?,
            (None, Some(u)) => writeln!(w, " {} <= {}", vname, u)?,
            (Some(l), None) => writeln!(w, " {} >= {}", vname, l)?,
            (Some(l), Some(u)) => writeln!(w, " {} <= {} <= {}", l, vname, u)?,
        }
    }
    w.flush()?;
    Ok(())
}

fn cartesian_product_ref(elements: &[Vec<String>]) -> Vec<Vec<usize>> {
    if elements.is_empty() { return vec![vec![]]; }
    let mut result = vec![vec![]];
    for dim in elements {
        let mut new_result = Vec::with_capacity(result.len() * dim.len());
        for combo in &result {
            for idx in 0..dim.len() {
                let mut new_combo = combo.clone();
                new_combo.push(idx);
                new_result.push(new_combo);
            }
        }
        result = new_result;
    }
    result
}

/// Write variable solution to CSV file
#[pyfunction]
fn write_var_solution_csv(
    filename: &str,
    dim_names: Vec<String>,
    dim_elements: Vec<Vec<String>>,
    values: PyReadonlyArray1<'_, f64>,
    duals: PyReadonlyArray1<'_, f64>,
) -> PyResult<()> {
    let vals = values.as_slice()?;
    let duals_slice = duals.as_slice()?;
    
    let file = File::create(filename)?;
    let mut w = BufWriter::with_capacity(1024 * 1024, file);
    
    // Header
    for (i, name) in dim_names.iter().enumerate() {
        if i > 0 { write!(w, ",")?; }
        write!(w, "{}", name)?;
    }
    writeln!(w, ",value,dual")?;
    
    // Data rows
    let combos = cartesian_product_ref(&dim_elements);
    for (i, combo) in combos.iter().enumerate() {
        for (d, &idx) in combo.iter().enumerate() {
            if d > 0 { write!(w, ",")?; }
            write!(w, "{}", &dim_elements[d][idx])?;
        }
        writeln!(w, ",{},{}", vals[i], duals_slice[i])?;
    }
    
    w.flush()?;
    Ok(())
}

/// Write numpy-compatible .npy file (float64 array)
#[pyfunction]
fn write_npy_f64(
    filename: &str,
    data: PyReadonlyArray1<'_, f64>,
) -> PyResult<()> {
    let slice = data.as_slice()?;
    let file = File::create(filename)?;
    let mut w = BufWriter::new(file);
    
    // NPY format header
    let header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}",
        slice.len()
    );
    let header_len = header.len();
    // Pad to 64-byte alignment
    let padding = 64 - ((10 + header_len) % 64);
    let total_header = header_len + padding;
    
    // Magic + version
    w.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y', 0x01, 0x00])?;
    // Header length (little-endian u16)
    w.write_all(&(total_header as u16).to_le_bytes())?;
    // Header + padding
    w.write_all(header.as_bytes())?;
    for _ in 0..padding-1 { w.write_all(b" ")?; }
    w.write_all(b"\n")?;
    
    // Data (little-endian f64)
    for &val in slice {
        w.write_all(&val.to_le_bytes())?;
    }
    
    w.flush()?;
    Ok(())
}

/// Write constraint solution to CSV file
#[pyfunction]
fn write_con_solution_csv(
    filename: &str,
    dim_names: Vec<String>,
    dim_elements: Vec<Vec<String>>,
    duals: PyReadonlyArray1<'_, f64>,
) -> PyResult<()> {
    let duals_slice = duals.as_slice()?;
    
    let file = File::create(filename)?;
    let mut w = BufWriter::with_capacity(64 * 1024, file);
    
    // Header
    for (i, name) in dim_names.iter().enumerate() {
        if i > 0 { write!(w, ",")?; }
        write!(w, "{}", name)?;
    }
    writeln!(w, ",dual")?;
    
    // Data rows
    if dim_elements.is_empty() {
        // Scalar constraint
        writeln!(w, "{}", duals_slice[0])?;
    } else {
        let combos = cartesian_product_ref(&dim_elements);
        for (i, combo) in combos.iter().enumerate() {
            for (d, &idx) in combo.iter().enumerate() {
                if d > 0 { write!(w, ",")?; }
                write!(w, "{}", &dim_elements[d][idx])?;
            }
            writeln!(w, ",{}", duals_slice[i])?;
        }
    }
    
    w.flush()?;
    Ok(())
}

/// Build CSR matrix for sum constraints - efficient version
/// 
/// Takes var_start_idx to compute final indices directly without string mapping.
/// This is the fast path for direct HiGHS solving.
///
/// Parameters:
/// - var_start_idx: starting index of this variable in the global variable array
/// - dim_sizes: size of each variable dimension (e.g., [2, 3] for 2x3 variable)
/// - is_free_dim: which dims iterate over constraints (true) vs sum within (false)
/// - coef_flat: flattened coefficient array (or None for all 1s)
/// - rhs_flat: RHS values, one per constraint
/// - sense: constraint sense ("<=" -> -inf to rhs, ">=" -> rhs to inf, "=" -> rhs to rhs)
///
/// Returns: (indptr, indices, data, row_lower, row_upper)
#[pyfunction]
#[pyo3(signature = (var_start_idx, dim_sizes, is_free_dim, coef_flat, rhs_flat, sense))]
fn build_sum_csr_fast(
    py: Python<'_>,
    var_start_idx: i32,
    dim_sizes: Vec<usize>,
    is_free_dim: Vec<bool>,
    coef_flat: Option<PyReadonlyArray1<'_, f64>>,
    rhs_flat: PyReadonlyArray1<'_, f64>,
    sense: &str,
) -> PyResult<(
    Py<PyArray1<i32>>,        // indptr
    Py<PyArray1<i32>>,        // indices  
    Py<PyArray1<f64>>,        // data
    Py<PyArray1<f64>>,        // row_lower
    Py<PyArray1<f64>>,        // row_upper
)> {
    let rhs_vals = rhs_flat.as_slice()?;
    let coef_data: Option<Vec<f64>> = coef_flat.map(|c| c.as_slice().unwrap().to_vec());
    
    let ndim = dim_sizes.len();
    
    // Compute strides for variable indexing (row-major order)
    let mut var_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        var_strides[i] = var_strides[i + 1] * dim_sizes[i + 1];
    }
    
    // Separate free (constraint) and sum dimensions
    let mut free_dims: Vec<usize> = Vec::new();
    let mut sum_dims: Vec<usize> = Vec::new();
    let mut n_free = 1usize;
    let mut n_sum = 1usize;
    
    for (i, &is_free) in is_free_dim.iter().enumerate() {
        if is_free {
            free_dims.push(i);
            n_free *= dim_sizes[i];
        } else {
            sum_dims.push(i);
            n_sum *= dim_sizes[i];
        }
    }
    
    let n_cons = n_free;
    let vars_per_con = n_sum;
    let nnz = n_cons * vars_per_con;
    
    // Pre-allocate CSR arrays
    let mut indptr = vec![0i32; n_cons + 1];
    let mut indices = vec![0i32; nnz];
    let mut data = vec![0f64; nnz];
    let mut row_lower = vec![0f64; n_cons];
    let mut row_upper = vec![0f64; n_cons];
    
    // Fill CSR arrays in parallel
    py.allow_threads(|| {
        // Build free dimension strides
        let mut free_strides = vec![1usize; free_dims.len()];
        for i in (0..free_dims.len().saturating_sub(1)).rev() {
            free_strides[i] = free_strides[i + 1] * dim_sizes[free_dims[i + 1]];
        }
        
        // Build sum dimension strides  
        let mut sum_strides = vec![1usize; sum_dims.len()];
        for i in (0..sum_dims.len().saturating_sub(1)).rev() {
            sum_strides[i] = sum_strides[i + 1] * dim_sizes[sum_dims[i + 1]];
        }
        
        (0..n_cons).into_par_iter().for_each(|con_idx| {
            // Decode con_idx into free dimension indices
            let mut free_idx = vec![0usize; free_dims.len()];
            let mut remaining = con_idx;
            for (fi, &stride) in free_strides.iter().enumerate() {
                free_idx[fi] = remaining / stride;
                remaining %= stride;
            }
            
            let row_start = con_idx * vars_per_con;
            
            for sum_idx in 0..vars_per_con {
                // Decode sum_idx into sum dimension indices
                let mut sum_indices = vec![0usize; sum_dims.len()];
                let mut rem = sum_idx;
                for (si, &stride) in sum_strides.iter().enumerate() {
                    sum_indices[si] = rem / stride;
                    rem %= stride;
                }
                
                // Build full index and compute flat variable index
                let mut full_idx = vec![0usize; ndim];
                let mut fi = 0;
                let mut si = 0;
                for d in 0..ndim {
                    if is_free_dim[d] {
                        full_idx[d] = free_idx[fi];
                        fi += 1;
                    } else {
                        full_idx[d] = sum_indices[si];
                        si += 1;
                    }
                }
                
                // Compute flat variable index using strides
                let mut var_flat_idx = 0usize;
                for (d, &idx) in full_idx.iter().enumerate() {
                    var_flat_idx += idx * var_strides[d];
                }
                
                // Get coefficient
                let c = if let Some(ref coef) = coef_data {
                    coef.get(var_flat_idx).copied().unwrap_or(1.0)
                } else {
                    1.0
                };
                
                // Store in CSR
                let pos = row_start + sum_idx;
                unsafe {
                    let indices_ptr = indices.as_ptr() as *mut i32;
                    let data_ptr = data.as_ptr() as *mut f64;
                    *indices_ptr.add(pos) = var_start_idx + var_flat_idx as i32;
                    *data_ptr.add(pos) = c;
                }
            }
        });
        
        // Fill indptr (sequential)
        for i in 0..=n_cons {
            indptr[i] = (i * vars_per_con) as i32;
        }
        
        // Fill bounds
        for (i, rhs) in rhs_vals.iter().enumerate() {
            match sense {
                "<=" => {
                    row_lower[i] = f64::NEG_INFINITY;
                    row_upper[i] = *rhs;
                }
                ">=" => {
                    row_lower[i] = *rhs;
                    row_upper[i] = f64::INFINITY;
                }
                _ => {
                    row_lower[i] = *rhs;
                    row_upper[i] = *rhs;
                }
            }
        }
    });
    
    Ok((
        indptr.to_pyarray_bound(py).into(),
        indices.to_pyarray_bound(py).into(),
        data.to_pyarray_bound(py).into(),
        row_lower.to_pyarray_bound(py).into(),
        row_upper.to_pyarray_bound(py).into(),
    ))
}

/// Build CSR matrix for sum constraints (e.g., Sum(j, x[i,j]) <= a[i])
/// 
/// This builds the constraint matrix directly in Rust, returning arrays
/// that can be passed to HiGHS without writing LP files.
///
/// Parameters:
/// - var_name: base variable name (e.g., "x")
/// - var_dim_elements: elements for each variable dimension
/// - is_free_dim: which dims iterate over constraints (true) vs sum within (false)
/// - coef_flat: flattened coefficient array (or None for all 1s)
/// - coef_shape: shape of coefficient array
/// - rhs_flat: RHS values, one per constraint
/// - sense: constraint sense ("<=" -> -inf to rhs, ">=" -> rhs to inf, "=" -> rhs to rhs)
///
/// Returns: (var_names, indptr, indices, data, row_lower, row_upper)
#[pyfunction]
#[pyo3(signature = (var_name, var_dim_elements, is_free_dim, coef_flat, coef_shape, rhs_flat, sense))]
fn build_sum_constraint_csr(
    py: Python<'_>,
    var_name: &str,
    var_dim_elements: Vec<Vec<String>>,
    is_free_dim: Vec<bool>,
    coef_flat: Option<PyReadonlyArray1<'_, f64>>,
    coef_shape: Vec<usize>,
    rhs_flat: PyReadonlyArray1<'_, f64>,
    sense: &str,
) -> PyResult<(
    Vec<String>,              // var_names
    Py<PyArray1<i32>>,        // indptr
    Py<PyArray1<i32>>,        // indices  
    Py<PyArray1<f64>>,        // data
    Py<PyArray1<f64>>,        // row_lower
    Py<PyArray1<f64>>,        // row_upper
)> {
    let rhs_vals = rhs_flat.as_slice()?;
    let coef_data: Option<Vec<f64>> = coef_flat.map(|c| c.as_slice().unwrap().to_vec());
    
    let ndim = var_dim_elements.len();
    
    // Compute strides for coefficient indexing
    let mut coef_strides = vec![1usize; coef_shape.len()];
    for i in (0..coef_shape.len().saturating_sub(1)).rev() {
        coef_strides[i] = coef_strides[i + 1] * coef_shape[i + 1];
    }
    
    // Separate free (constraint) and sum dimensions
    let mut free_dims: Vec<usize> = Vec::new();
    let mut sum_dims: Vec<usize> = Vec::new();
    for (i, &is_free) in is_free_dim.iter().enumerate() {
        if is_free {
            free_dims.push(i);
        } else {
            sum_dims.push(i);
        }
    }
    
    // Generate all variable names
    let all_combos = cartesian_product_ref(&var_dim_elements);
    let n_vars = all_combos.len();
    let var_names: Vec<String> = all_combos.iter().map(|combo| {
        let mut name = var_name.to_string();
        for (d, &idx) in combo.iter().enumerate() {
            name.push('_');
            name.push_str(&var_dim_elements[d][idx]);
        }
        name
    }).collect();
    
    // Build variable index map (combo -> flat index)
    let mut var_idx_map: std::collections::HashMap<Vec<usize>, usize> = 
        std::collections::HashMap::with_capacity(n_vars);
    for (i, combo) in all_combos.iter().enumerate() {
        var_idx_map.insert(combo.clone(), i);
    }
    
    // Generate free (constraint) combinations
    let free_elements: Vec<Vec<String>> = free_dims.iter()
        .map(|&d| var_dim_elements[d].clone())
        .collect();
    let free_combos = cartesian_product_ref(&free_elements);
    let n_cons = free_combos.len();
    
    // Generate sum combinations
    let sum_elements: Vec<&Vec<String>> = sum_dims.iter()
        .map(|&d| &var_dim_elements[d])
        .collect();
    let sum_combos = cartesian_product(&sum_elements);
    let vars_per_con = sum_combos.len();
    
    // Pre-allocate CSR arrays
    let nnz = n_cons * vars_per_con;
    let mut indptr = vec![0i32; n_cons + 1];
    let mut indices = vec![0i32; nnz];
    let mut data = vec![0f64; nnz];
    let mut row_lower = vec![0f64; n_cons];
    let mut row_upper = vec![0f64; n_cons];
    
    // Fill CSR arrays in parallel
    py.allow_threads(|| {
        free_combos.par_iter().enumerate().for_each(|(con_idx, free_combo)| {
            let row_start = con_idx * vars_per_con;
            
            for (j, sum_combo) in sum_combos.iter().enumerate() {
                // Build full index in variable dimension order
                let mut full_idx = vec![0usize; ndim];
                let mut fi = 0;
                let mut si = 0;
                for d in 0..ndim {
                    if is_free_dim[d] {
                        full_idx[d] = free_combo[fi];
                        fi += 1;
                    } else {
                        full_idx[d] = sum_combo[si];
                        si += 1;
                    }
                }
                
                // Get variable index
                let var_idx = *var_idx_map.get(&full_idx).unwrap();
                
                // Get coefficient
                let c = if let Some(ref coef) = coef_data {
                    let mut flat_idx = 0;
                    for (d, &idx) in full_idx.iter().enumerate() {
                        if d < coef_strides.len() {
                            flat_idx += idx * coef_strides[d];
                        }
                    }
                    coef.get(flat_idx).copied().unwrap_or(1.0)
                } else {
                    1.0
                };
                
                // Store in CSR (note: indices/data are pre-sized, so direct write)
                let pos = row_start + j;
                // SAFETY: We pre-sized the arrays and each thread writes to disjoint positions
                unsafe {
                    let indices_ptr = indices.as_ptr() as *mut i32;
                    let data_ptr = data.as_ptr() as *mut f64;
                    *indices_ptr.add(pos) = var_idx as i32;
                    *data_ptr.add(pos) = c;
                }
            }
        });
        
        // Fill indptr and bounds (sequential, fast)
        for i in 0..=n_cons {
            indptr[i] = (i * vars_per_con) as i32;
        }
        
        for (i, rhs) in rhs_vals.iter().enumerate() {
            match sense {
                "<=" => {
                    row_lower[i] = f64::NEG_INFINITY;
                    row_upper[i] = *rhs;
                }
                ">=" => {
                    row_lower[i] = *rhs;
                    row_upper[i] = f64::INFINITY;
                }
                _ => {
                    row_lower[i] = *rhs;
                    row_upper[i] = *rhs;
                }
            }
        }
    });
    
    Ok((
        var_names,
        indptr.to_pyarray_bound(py).into(),
        indices.to_pyarray_bound(py).into(),
        data.to_pyarray_bound(py).into(),
        row_lower.to_pyarray_bound(py).into(),
        row_upper.to_pyarray_bound(py).into(),
    ))
}

/// Generate variable names efficiently in Rust
/// Returns list of names like ["x_a_1", "x_a_2", "x_b_1", "x_b_2"]
#[pyfunction]
fn generate_var_names(
    py: Python<'_>,
    var_name: &str,
    dim_elements: Vec<Vec<String>>,
) -> PyResult<Vec<String>> {
    let names: Vec<String> = py.allow_threads(|| {
        let combos = cartesian_product_ref(&dim_elements);
        combos.into_par_iter().map(|combo| {
            let mut name = var_name.to_string();
            for (d, idx) in combo.iter().enumerate() {
                name.push('_');
                name.push_str(&dim_elements[d][*idx]);
            }
            name
        }).collect()
    });
    Ok(names)
}

/// Write multi-term constraints with variable names generated in Rust.
/// 
/// Each term is specified by (var_name, dim_elements, coef_flat, is_free_dim).
/// For each constraint row (determined by free_sets), we build all variable names
/// and collect their coefficients.
///
/// term_specs: Vec of (var_name, dim_elements, coef_flat, is_free_dim) for each term
/// free_set_sizes: sizes of the free sets (determines number of constraints)
/// rhs_flat: RHS values, one per constraint
#[pyfunction]
#[pyo3(signature = (filename, con_name, sense, term_var_names, term_dim_elements, term_coefs, term_coef_shapes, term_coef_free_maps, term_coef_sum_maps, term_var_free_maps, free_set_sizes, rhs_flat, term_scalar_coefs))]
fn write_multi_term_constraints(
    py: Python<'_>,
    filename: &str,
    con_name: &str,
    sense: &str,
    term_var_names: Vec<String>,
    term_dim_elements: Vec<Vec<Vec<String>>>,
    term_coefs: Vec<Option<PyReadonlyArray1<'_, f64>>>,
    term_coef_shapes: Vec<Vec<usize>>,
    term_coef_free_maps: Vec<Vec<i32>>,
    term_coef_sum_maps: Vec<Vec<i32>>,
    term_var_free_maps: Vec<Vec<i32>>,
    free_set_sizes: Vec<usize>,
    rhs_flat: PyReadonlyArray1<'_, f64>,
    term_scalar_coefs: Vec<f64>,
) -> PyResult<()> {
    let rhs_vals = rhs_flat.as_slice()?;
    let sense_str = match sense { "<=" => "<=", ">=" => ">=", _ => "=" };
    let n_terms = term_var_names.len();
    
    // Convert coefficient arrays to owned Vecs
    let coef_data: Vec<Option<Vec<f64>>> = term_coefs.into_iter()
        .map(|opt| opt.map(|arr| arr.as_slice().unwrap().to_vec()))
        .collect();
    
    // Calculate total number of constraints
    let n_cons: usize = free_set_sizes.iter().product();
    
    // Precompute strides for coefficient arrays based on their own shapes
    let term_coef_strides: Vec<Vec<usize>> = term_coef_shapes.iter()
        .map(|shape| {
            let mut strides = vec![1usize; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            strides
        })
        .collect();
    
    // Compute free set strides for decoding constraint index
    let mut free_strides = vec![1usize; free_set_sizes.len()];
    for i in (0..free_set_sizes.len().saturating_sub(1)).rev() {
        free_strides[i] = free_strides[i + 1] * free_set_sizes[i + 1];
    }
    
    let lines: Vec<String> = py.allow_threads(|| {
        (0..n_cons).into_par_iter().map(|con_idx| {
            // Decode con_idx into free dimension indices
            let mut free_indices = vec![0usize; free_set_sizes.len()];
            let mut remaining = con_idx;
            for (fi, &stride) in free_strides.iter().enumerate() {
                free_indices[fi] = remaining / stride;
                remaining %= stride;
            }
            
            let mut s = format!(" {}_{}: ", con_name, con_idx);
            let mut first = true;
            
            // Process each term
            for t in 0..n_terms {
                let var_name = &term_var_names[t];
                let dims = &term_dim_elements[t];
                let var_free_map = &term_var_free_maps[t];
                let coef_strides = &term_coef_strides[t];
                let coef_free_map = &term_coef_free_maps[t];
                let coef_sum_map = &term_coef_sum_maps[t];

                // Determine which dim indices are fixed (bound to a free set) vs
                // summed. Map each free dim to its free set BY IDENTITY via
                // var_free_map[d] (the free-set index, or -1 when summed) - a
                // positional counter wrongly assumes the variable's free dims
                // line up with the constraint's free sets in order.
                let mut fixed_dims: Vec<(usize, usize)> = Vec::new();
                let mut sum_dims: Vec<usize> = Vec::new();

                for (d, &fs) in var_free_map.iter().enumerate() {
                    if fs >= 0 {
                        fixed_dims.push((d, free_indices[fs as usize]));
                    } else {
                        sum_dims.push(d);
                    }
                }
                
                // Generate all combinations of summed dimensions
                let sum_combos = if sum_dims.is_empty() {
                    vec![vec![]]
                } else {
                    let sum_sizes: Vec<usize> = sum_dims.iter().map(|&d| dims[d].len()).collect();
                    cartesian_indices(&sum_sizes)
                };
                
                for sum_combo in sum_combos {
                    // Build full index for variable
                    let mut full_idx = vec![0usize; dims.len()];
                    for &(d, idx) in &fixed_dims {
                        full_idx[d] = idx;
                    }
                    for (si, &d) in sum_dims.iter().enumerate() {
                        full_idx[d] = sum_combo[si];
                    }
                    
                    // Build variable name
                    let mut vname = var_name.clone();
                    for (d, &idx) in full_idx.iter().enumerate() {
                        vname.push('_');
                        vname.push_str(&dims[d][idx]);
                    }
                    
                    // Get coefficient. Each coef dim is either a free axis
                    // (indexed by free_indices) or one of the variable's summed
                    // axes (indexed by sum_combo). Ignoring the summed axes -
                    // the old bug - turned Sum(j, a[i,j]*x[j]) into
                    // a[i,0]*Sum(j, x[j]).
                    let c = if let Some(ref coef) = coef_data[t] {
                        let mut flat_idx = 0usize;
                        for (cd, &free_set_idx) in coef_free_map.iter().enumerate() {
                            if free_set_idx >= 0 {
                                flat_idx += free_indices[free_set_idx as usize] * coef_strides[cd];
                            } else {
                                let sum_pos = coef_sum_map[cd];
                                if sum_pos >= 0 {
                                    flat_idx += sum_combo[sum_pos as usize] * coef_strides[cd];
                                }
                            }
                        }
                        coef.get(flat_idx).copied().unwrap_or(1.0)
                    } else {
                        term_scalar_coefs[t]
                    };
                    
                    if c != 0.0 {
                        append_term(&mut s, c, &vname, first);
                        first = false;
                    }
                }
            }
            
            if first { s.push_str("0"); }
            let rval = rhs_vals.get(con_idx).copied().unwrap_or(0.0);
            s.push_str(&format!(" {} {}\n", sense_str, rval));
            s
        }).collect()
    });
    
    let file = File::options().append(true).open(filename)?;
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, file);
    for line in lines { w.write_all(line.as_bytes())?; }
    w.flush()?;
    Ok(())
}

fn cartesian_indices(sizes: &[usize]) -> Vec<Vec<usize>> {
    if sizes.is_empty() { return vec![vec![]]; }
    let mut result = vec![vec![]];
    for &size in sizes {
        let mut new_result = Vec::with_capacity(result.len() * size);
        for combo in &result {
            for idx in 0..size {
                let mut new_combo = combo.clone();
                new_combo.push(idx);
                new_result.push(new_combo);
            }
        }
        result = new_result;
    }
    result
}

/// Build CSR matrix for multi-term constraints (e.g., G[T,H] - N[T] <= 0)
///
/// Each term is specified by its var_start_idx, dim_sizes, coef, and is_free_dim.
/// This handles multiple variables in the same constraint row.
///
/// Parameters:
/// - term_var_starts: starting index in global variable array for each term
/// - term_dim_sizes: list of dimension sizes for each term
/// - term_coefs: flattened coefficients for each term (None = all 1s)
/// - term_coef_sizes: shape of coefficient array for each term (for proper indexing)
/// - term_coef_free_map: for each term, maps coef dimension to free set index (-1 if not mapped)
/// - term_is_free_dims: which dims are free (iterate constraints) vs sum
/// - free_set_sizes: sizes of the free sets (determines number of constraints)
/// - rhs_flat: RHS values, one per constraint
/// - sense: constraint sense
///
/// Returns: (indptr, indices, data, row_lower, row_upper)
#[pyfunction]
#[pyo3(signature = (term_var_starts, term_dim_sizes, term_coefs, term_coef_sizes, term_coef_free_map, term_coef_sum_map, term_var_free_maps, free_set_sizes, rhs_flat, sense))]
fn build_multi_term_csr(
    py: Python<'_>,
    term_var_starts: Vec<i32>,
    term_dim_sizes: Vec<Vec<usize>>,
    term_coefs: Vec<Option<PyReadonlyArray1<'_, f64>>>,
    term_coef_sizes: Vec<Vec<usize>>,      // shape of coef array for each term
    term_coef_free_map: Vec<Vec<i32>>,     // coef dim -> free set index (-1 if not a free dim)
    term_coef_sum_map: Vec<Vec<i32>>,      // coef dim -> variable summed-axis position (-1 if not a summed dim)
    term_var_free_maps: Vec<Vec<i32>>,     // var dim -> free set index (-1 if summed)
    free_set_sizes: Vec<usize>,
    rhs_flat: PyReadonlyArray1<'_, f64>,
    sense: &str,
) -> PyResult<(
    Py<PyArray1<i32>>,        // indptr
    Py<PyArray1<i32>>,        // indices
    Py<PyArray1<f64>>,        // data
    Py<PyArray1<f64>>,        // row_lower
    Py<PyArray1<f64>>,        // row_upper
)> {
    let rhs_vals = rhs_flat.as_slice()?;
    let n_terms = term_var_starts.len();

    // Convert coefficient arrays to owned Vecs
    let coef_data: Vec<Option<Vec<f64>>> = term_coefs.into_iter()
        .map(|opt| opt.map(|arr| arr.as_slice().unwrap().to_vec()))
        .collect();

    // Calculate total number of constraints
    let n_cons: usize = free_set_sizes.iter().product();

    // Precompute strides for each term's variable dimensions
    let term_var_strides: Vec<Vec<usize>> = term_dim_sizes.iter()
        .map(|sizes| {
            let mut strides = vec![1usize; sizes.len()];
            for i in (0..sizes.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * sizes[i + 1];
            }
            strides
        })
        .collect();

    // Precompute strides for each term's coefficient dimensions
    let term_coef_strides: Vec<Vec<usize>> = term_coef_sizes.iter()
        .map(|sizes| {
            let mut strides = vec![1usize; sizes.len()];
            for i in (0..sizes.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * sizes[i + 1];
            }
            strides
        })
        .collect();

    // Compute free set strides for decoding constraint index
    let mut free_strides = vec![1usize; free_set_sizes.len()];
    for i in (0..free_set_sizes.len().saturating_sub(1)).rev() {
        free_strides[i] = free_strides[i + 1] * free_set_sizes[i + 1];
    }

    // Count total non-zeros per row (for pre-allocation)
    let mut nnz_per_row = vec![0usize; n_cons];
    for t in 0..n_terms {
        let var_free_map = &term_var_free_maps[t];
        let sizes = &term_dim_sizes[t];
        let mut sum_count = 1usize;
        for (d, &fs) in var_free_map.iter().enumerate() {
            if fs < 0 {
                sum_count *= sizes[d];
            }
        }
        for row in &mut nnz_per_row {
            *row += sum_count;
        }
    }

    let total_nnz: usize = nnz_per_row.iter().sum();

    // Pre-allocate CSR arrays
    let mut indptr = vec![0i32; n_cons + 1];
    let mut indices = vec![0i32; total_nnz];
    let mut data = vec![0f64; total_nnz];
    let mut row_lower = vec![0f64; n_cons];
    let mut row_upper = vec![0f64; n_cons];

    // Build indptr
    for i in 0..n_cons {
        indptr[i + 1] = indptr[i] + nnz_per_row[i] as i32;
    }

    // Fill CSR in parallel
    py.allow_threads(|| {
        (0..n_cons).into_par_iter().for_each(|con_idx| {
            // Decode con_idx into free dimension indices
            let mut free_indices = vec![0usize; free_set_sizes.len()];
            let mut remaining = con_idx;
            for (fi, &stride) in free_strides.iter().enumerate() {
                free_indices[fi] = remaining / stride;
                remaining %= stride;
            }

            let row_start = indptr[con_idx] as usize;
            let mut pos = row_start;

            // Process each term
            for t in 0..n_terms {
                let var_start = term_var_starts[t];
                let sizes = &term_dim_sizes[t];
                let var_free_map = &term_var_free_maps[t];
                let var_strides = &term_var_strides[t];

                // Separate fixed (bound to a free set) and summed dimensions.
                // Map each free dim to its free set BY IDENTITY via
                // var_free_map[d]; a positional counter wrongly assumes the
                // variable's free dims match the constraint's free sets in order.
                let mut sum_dims: Vec<usize> = Vec::new();
                let mut fixed_vals: Vec<(usize, usize)> = Vec::new();
                for (d, &fs) in var_free_map.iter().enumerate() {
                    if fs >= 0 {
                        fixed_vals.push((d, free_indices[fs as usize]));
                    } else {
                        sum_dims.push(d);
                    }
                }

                // Generate all combinations of summed dimensions
                let sum_combos = if sum_dims.is_empty() {
                    vec![vec![]]
                } else {
                    let sum_sizes: Vec<usize> = sum_dims.iter().map(|&d| sizes[d]).collect();
                    cartesian_indices(&sum_sizes)
                };

                for sum_combo in sum_combos {
                    // Build full index
                    let mut full_idx = vec![0usize; sizes.len()];
                    for &(d, idx) in &fixed_vals {
                        full_idx[d] = idx;
                    }
                    for (si, &d) in sum_dims.iter().enumerate() {
                        full_idx[d] = sum_combo[si];
                    }

                    // Compute flat variable index
                    let mut var_flat_idx = 0usize;
                    for (d, &idx) in full_idx.iter().enumerate() {
                        var_flat_idx += idx * var_strides[d];
                    }

                    // Compute coefficient index. Each coef dim is either a free
                    // axis (indexed by the constraint's free_indices) or one of
                    // the variable's summed axes (indexed by sum_combo). Pinning a
                    // summed axis to 0 -- the old bug -- turned Sum(j, a[i,j]*x[j])
                    // into a[i,0]*Sum(j, x[j]).
                    let c = if let Some(ref coef) = coef_data[t] {
                        let coef_free_map = &term_coef_free_map[t];
                        let coef_sum_map = &term_coef_sum_map[t];
                        let coef_strides = &term_coef_strides[t];

                        let mut coef_flat_idx = 0usize;
                        for (cd, &free_idx) in coef_free_map.iter().enumerate() {
                            if free_idx >= 0 {
                                coef_flat_idx += free_indices[free_idx as usize] * coef_strides[cd];
                            } else {
                                let sum_pos = coef_sum_map[cd];
                                if sum_pos >= 0 {
                                    coef_flat_idx += sum_combo[sum_pos as usize] * coef_strides[cd];
                                }
                            }
                        }
                        coef.get(coef_flat_idx).copied().unwrap_or(1.0)
                    } else {
                        1.0
                    };

                    // Always write entry (including zeros) since we pre-allocated
                    // based on structural sparsity. Solver handles explicit zeros.
                    unsafe {
                        let indices_ptr = indices.as_ptr() as *mut i32;
                        let data_ptr = data.as_ptr() as *mut f64;
                        *indices_ptr.add(pos) = var_start + var_flat_idx as i32;
                        *data_ptr.add(pos) = c;
                    }
                    pos += 1;
                }
            }
        });

        // Fill bounds
        for (i, rhs) in rhs_vals.iter().enumerate() {
            match sense {
                "<=" => {
                    row_lower[i] = f64::NEG_INFINITY;
                    row_upper[i] = *rhs;
                }
                ">=" => {
                    row_lower[i] = *rhs;
                    row_upper[i] = f64::INFINITY;
                }
                _ => {
                    row_lower[i] = *rhs;
                    row_upper[i] = *rhs;
                }
            }
        }
    });

    Ok((
        indptr.to_pyarray_bound(py).into(),
        indices.to_pyarray_bound(py).into(),
        data.to_pyarray_bound(py).into(),
        row_lower.to_pyarray_bound(py).into(),
        row_upper.to_pyarray_bound(py).into(),
    ))
}

#[pymodule]
fn nimopt_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(write_lp_header, m)?)?;
    m.add_function(wrap_pyfunction!(write_objective, m)?)?;
    m.add_function(wrap_pyfunction!(write_subject_to, m)?)?;
    m.add_function(wrap_pyfunction!(write_uniform_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(write_coef_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(write_sum_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(write_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(write_var_types, m)?)?;
    m.add_function(wrap_pyfunction!(write_single_constraint, m)?)?;
    m.add_function(wrap_pyfunction!(write_bounds_fast, m)?)?;
    m.add_function(wrap_pyfunction!(write_objective_fast, m)?)?;
    m.add_function(wrap_pyfunction!(write_var_solution_csv, m)?)?;
    m.add_function(wrap_pyfunction!(write_con_solution_csv, m)?)?;
    m.add_function(wrap_pyfunction!(write_npy_f64, m)?)?;
    m.add_function(wrap_pyfunction!(build_sum_constraint_csr, m)?)?;
    m.add_function(wrap_pyfunction!(build_sum_csr_fast, m)?)?;
    m.add_function(wrap_pyfunction!(generate_var_names, m)?)?;
    m.add_function(wrap_pyfunction!(write_multi_term_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(build_multi_term_csr, m)?)?;
    Ok(())
}
