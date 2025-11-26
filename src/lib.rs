use anyhow::Result;
use spin_sdk::http::{IntoResponse, Request, Response};
use spin_sdk::http_component;
use std::io::BufRead;
use std::io::Cursor;
use std::vec;
use tract_tensorflow::prelude::*;

/// A simple Spin HTTP component.
#[http_component]
fn handle_spin_image_classification_demo(req: Request) -> anyhow::Result<impl IntoResponse> {
    let image = req.body().to_vec();

    if image.is_empty() {
        return Ok(Response::builder()
            .status(400)
            .header("content-type", "text/plain")
            .body("No image data received.".to_string())
            .build());
    }

    println!(
        "[Rust classifier]: Received image with {} bytes.",
        image.len()
    );

    let classification_result = classify(image);

    if classification_result.is_err() {
        eprintln!(
            "[Rust classifier]: Error during classification: {:?}",
            classification_result.err()
        );

        return Ok(Response::builder()
            .status(500)
            .header("content-type", "text/plain")
            .body(format!("Error during classification",))
            .build());
    }

    // If we have a successful classification, return the result.
    let (label, probability) = classification_result.unwrap();
    let body = format!(
        "{{\"Predicted label\": \"{}\", \"Probability\": {:.4}}}",
        label, probability
    );
    Ok(Response::builder()
        .status(200)
        .header("content-type", "text/plain")
        .body(body)
        .build())
}

type ClassificationResult = (String, f32);

#[derive(Debug)]
enum ClassificationError {
    ModelError(String),
    ImageError(String),
    IoError(String),
    Unknown(String),
    Unclassified,
}

fn classify(img: Vec<u8>) -> Result<ClassificationResult, ClassificationError> {
    let model = tract_tensorflow::tensorflow()
        .model_for_read(&mut Cursor::new(include_bytes!(
            "../mobilenet_v2_1.4_224_frozen.pb"
        )))?
        .with_input_fact(0, f32::fact([1, 224, 224, 3]).into())?
        .into_optimized()?
        .into_runnable()?;

    println!("[Rust classifier]: Loaded Tensorflow model.");

    let image = image::load_from_memory(&img)?.to_rgb8();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into();

    println!("[Rust classifier]: Resized image to 224x224 px.");

    // run the model on the input
    let result = model.run(tvec!(image.into()))?;
    // find and display the max value with its index
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    match best {
        Some((probability, class)) => {
            let label = get_label(class)?;
            println!(
                "[Rust classifier]: Probability: {}, class: {}.",
                probability, label
            );
            return Ok((label, probability));
        }
        None => return Err(ClassificationError::Unclassified),
    }
}

fn get_label(num: usize) -> Result<String, anyhow::Error> {
    // The result of executing the inference is the predicted class,
    // which also indicates the line number in the (1-indexed) labels file.
    let labels = include_bytes!("../labels.txt");
    let content = std::io::BufReader::new(Cursor::new(labels));
    content
        .lines()
        .nth(num - 1)
        .expect("cannot get prediction label")
        .map_err(|err| anyhow::Error::new(err))
}

impl From<TractError> for ClassificationError {
    fn from(e: TractError) -> Self {
        ClassificationError::ModelError(e.to_string())
    }
}

impl From<image::ImageError> for ClassificationError {
    fn from(e: image::ImageError) -> Self {
        ClassificationError::ImageError(e.to_string())
    }
}
