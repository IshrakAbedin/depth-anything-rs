use anyhow::Result;
use clap::Parser;

use depth_anything_rs::{args::Args, run_depth_anything};

fn main() -> Result<()> {
    let args = Args::parse();
    run_depth_anything(args)?;
    Ok(())
}
