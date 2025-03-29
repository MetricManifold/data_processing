use std::env;
use std;
use regex::Regex;

mod server;
mod read;
mod compute;
mod msd;
mod plot;
mod utils;

#[tokio::main]
async fn main() {
    let routes = server::build_server();

    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}

fn cli() {
    // Get the command-line arguments
    let args: Vec<String> = env::args().collect();

    // Print the command-line arguments
    for (index, arg) in args.iter().enumerate() {
        println!("Argument {}: {}", index, arg);
    }

    let re = match Regex::new(r"compute:(\w+)") {
        Ok(re) => re,
        Err(e) => {
            eprintln!("Error creating regex: {}", e);
            return;
        }
    };

    let command = &args[1];

    if command == "results" {
        match msd::compile_results() {
            Ok(_) => println!("Successfully compiled results"),
            Err(e) => eprintln!("Error compiling results: {}", e),
        };
    } else if command == "plot_com" {
        match msd::plot_coms() {
            Ok(_) => println!("Successfully plotted center of mass"),
            Err(e) => eprintln!("Error plotting center of mass: {}", e),
        }
    } else if let Some(captures) = re.captures(command) {
        if let Some(compute_arg) = captures.get(1) {
            if compute_arg.as_str() == "msd" {
                match msd::compute_msd() {
                    Ok(_) => println!("Successfully computed mean squared displacement"),
                    Err(e) => eprintln!("Error computing mean squared displacement: {}", e),
                }
                println!("Computing mean squared displacement");
            } else {
                eprintln!("Unknown compute argument: {}", compute_arg.as_str());
            }
        }
    }
}
