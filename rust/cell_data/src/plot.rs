use plotters::prelude::*;
use std::error::Error;

pub fn plot_center_of_mass(
    com: &[(i32, u32, (f64, f64))],
    output_dir: &str
) -> Result<(), Box<dyn Error>> {
    let output_dir_str = format!("{}/center_of_mass.png", output_dir);
    let root = BitMapBackend::new(&output_dir_str, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Center of Mass", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..1600.0, 0.0..1600.0)?;

    chart.configure_mesh().draw()?;

    for &(_, _, (x, y)) in com {
        chart.draw_series(
            PointSeries::of_element(
                vec![(
                    if x > 1600.0 { x - 1600.0 } else { x },
                    if y > 1600.0 { y - 1600.0 } else { y },
                )],
                1,
                &RED,
                &(|c, s, st| {
                    EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
                    // +Text::new(format!("{}-{}", time_step, index), (10, 0), ("sans-serif", 15))
                })
            )
        )?;
    }

    root.present()?;
    println!("Center of mass plot saved to {}", output_dir_str);
    Ok(())
}

pub fn plot_msd(msd: &[(i32, f64)], output_dir: &str) -> Result<(), Box<dyn Error>> {
    let output_dir_str = format!("{}/msd.png", output_dir);
    let root = BitMapBackend::new(&output_dir_str, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("MSD Plot", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            msd
                .iter()
                .map(|(x, _)| *x)
                .min()
                .unwrap()..msd
                .iter()
                .map(|(x, _)| *x)
                .max()
                .unwrap(),
            msd
                .iter()
                .map(|(_, y)| *y)
                .into_iter()
                .reduce(f64::min)
                .unwrap()..msd
                .iter()
                .map(|(_, y)| *y)
                .into_iter()
                .reduce(f64::max)
                .unwrap()
        )?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        PointSeries::of_element(
            msd.into_iter().map(|(x, y)| (*x, *y)),
            2,
            &RED,
            &(|c, s, st| {
                EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
                // +Text::new(format!("{}-{}", time_step, index), (10, 0), ("sans-serif", 15))
            })
        )
    )?;

    root.present()?;
    println!("Data plot saved to {}", output_dir_str);
    Ok(())
}
