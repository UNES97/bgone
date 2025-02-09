import { AutoModel, AutoProcessor, RawImage } from "@huggingface/transformers";

const status = document.getElementById("status");
const fileUpload = document.getElementById("upload");
const uploadLabel = document.querySelector(".upload-label");
const imageContainer = document.querySelector(".image-container");
const canvas = document.getElementById("canvas");
const loader = document.querySelector(".loader");
const example = document.getElementById("example");

let model, processor;

// Load model and processor
status.textContent = "Loading model...";
loader.style.display = "block";

async function initialize() {
    model = await AutoModel.from_pretrained("briaai/RMBG-1.4", {
        config: { model_type: "custom" },
    });

    processor = await AutoProcessor.from_pretrained("briaai/RMBG-1.4", {
        config: {
            do_normalize: true,
            do_pad: false,
            do_rescale: true,
            do_resize: true,
            image_mean: [0.5, 0.5, 0.5],
            feature_extractor_type: "ImageFeatureExtractor",
            image_std: [1, 1, 1],
            resample: 2,
            rescale_factor: 0.00392156862745098,
            size: { width: 1024, height: 1024 },
        },
    });

    status.textContent = "Ready to upload an image.";
    loader.style.display = "none";
}

initialize();

fileUpload.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (!file) return;

    // Disable the upload button
    uploadLabel.classList.add("disabled");
    uploadLabel.style.pointerEvents = "none";

    const reader = new FileReader();
    reader.onload = (e2) => predict(e2.target.result);
    reader.readAsDataURL(file);
});

// Predict foreground of the given image
async function predict(url) {
    // Show loader and update status
    status.textContent = "Processing image...";
    loader.style.display = "block";
    canvas.style.display = "none";

    // Read image
    const image = await RawImage.fromURL(url);

    // Preprocess image
    const { pixel_values } = await processor(image);

    // Predict alpha matte
    const { output } = await model({ input: pixel_values });

    // Resize mask back to original size
    const mask = await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(
        image.width,
        image.height,
    );
    image.putAlpha(mask);

    // Draw image on canvas
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image.toCanvas(), 0, 0);

    // Update UI
    canvas.style.display = "block";
    loader.style.display = "none";
    status.textContent = "Background removed successfully!";

    // Re-enable the upload button after processing
    uploadLabel.classList.remove("disabled");
    uploadLabel.style.pointerEvents = "auto";
}