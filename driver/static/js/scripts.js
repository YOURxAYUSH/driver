document.addEventListener("DOMContentLoaded", () => {
    // Fetch GPS data periodically
    setInterval(() => {
        fetch("/gps")
            .then(response => response.json())
            .then(data => {
                document.getElementById("gps-latitude").textContent = data.latitude;
                document.getElementById("gps-longitude").textContent = data.longitude;
                document.getElementById("car-speed").textContent = data.speed;
            })
            .catch(err => console.error("Error fetching GPS data:", err));
    }, 5000); // Update every 5 seconds

    setInterval(() => {
        fetch("/user_info")
            .then(response => response.json())
            .then(data => {
                document.getElementById("user-status").textContent = data.status;
                const userInfo = data.user_info;

                if (userInfo && typeof userInfo === "object") {
                    document.getElementById("user-name").textContent = userInfo.name || "Unknown";
                    document.getElementById("user-license").textContent = userInfo.license || "N/A";
                    document.getElementById("user-vehicle").textContent = userInfo.vehicle || "N/A";

                    if (data.image_path) {
                        document.getElementById("user-image").src = data.image_path;
                    } else {
                        document.getElementById("user-image").src = "/static/default_face.jpg";
                    }
                } else {
                    document.getElementById("user-name").textContent = userInfo;
                    document.getElementById("user-license").textContent = "N/A";
                    document.getElementById("user-vehicle").textContent = "N/A";
                    document.getElementById("user-image").src = "/static/default_face.jpg";
                }
            })
            .catch(err => console.error("Error fetching user info:", err));
    }, 3000); // Poll every 3 seconds
});
