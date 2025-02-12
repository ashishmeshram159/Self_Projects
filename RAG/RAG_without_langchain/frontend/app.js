
// Use this to run up the http server for front end:
// python -m http.server 3000

async function uploadPDF() {
    const fileInput = document.getElementById('pdfUpload');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a file to upload.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://127.0.0.1:8000/upload/', {
            method: 'POST',
            body: formData,  
            mode: 'cors',  
        });

        const result = await response.json();
        if (response.ok) {
            alert(result.message);  
        } else {
            console.error(result);
            alert(`Error: ${result.detail || 'File upload failed'}`);
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while uploading the file.");
    }
}


async function askQuery() {
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value;

    if (!query) {
        alert("Please enter a query.");
        return;
    }

    try {
        const response = await fetch('http://127.0.0.1:8000/ask/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
            mode: 'cors',  // Enable CORS
        });

        const result = await response.json();
        const responseDiv = document.getElementById('response');
        if (response.ok) {
            responseDiv.innerHTML = `<strong>Response:</strong> ${result.message}`;
        } else {
            console.error(result);
            responseDiv.innerHTML = `<strong>Error:</strong> ${result.detail || 'Query failed'}`;
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while processing the query.");
    }
}