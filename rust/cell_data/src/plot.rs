use plotters::prelude::*;
use std::error::Error;

pub fn plot_center_of_mass(
    com: &[(i32, u32, (f64, f64))],
    output_dir: &str
) -> Result<(), Box<dyn Error>> {
    let output_dir_str = format!("{}/center_of_mass.png", output_dir);
    let root = BitMapBackend::new("center_of_mass.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Center of Mass", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..1600.0, 0.0..1600.0)?;

    chart.configure_mesh().draw()?;

    for &(time_step, index, (x, y)) in com {
        chart.draw_series(
            PointSeries::of_element(
                vec![(x, y)],
                5,
                &RED,
                &(|c, s, st| {
                    return (
                        (
                            EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
                            // +Text::new(format!("{}-{}", time_step, index), (10, 0), ("sans-serif", 15))
                        )
                    );
                })
            )
        )?;
    }

    root.present()?;
    Ok(())
}
