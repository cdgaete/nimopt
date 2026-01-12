//! Rust accelerated LP writer for nimopt

use numpy::PyReadonlyArray1;
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
    Ok(())
}
