document.getElementById("submit-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const loading = document.getElementById("loading");
    const results = document.getElementById("results");
    const error = document.getElementById("error");

    // Reset state
    results.style.display = "none";
    error.style.display = "none";
    loading.style.display = "block";

    try {
        const formData = new FormData(e.target);
        const res = await fetch("/submit", { method: "POST", body: formData });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);

        const { heatmap, overlaid } = await res.json();

        document.getElementById("result-image1").src = `data:image/jpeg;base64,${heatmap}`;
        document.getElementById("result-image2").src = `data:image/jpeg;base64,${overlaid}`;

        results.style.display = "block";
    } catch (err) {
        error.textContent = err.message;
        error.style.display = "block";
    } finally {
        loading.style.display = "none";
    }
});
