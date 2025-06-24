document.addEventListener("DOMContentLoaded", (event) => {
  const video = document.getElementById("video");
  const captureButton = document.getElementById("capture");
  const switchCameraButton = document.getElementById("switchCamera");
  const debugButton = document.getElementById("debug");
  const canvas = document.getElementById("canvas");
  const resultsDiv = document.getElementById("results");
  const loadingDiv = document.getElementById("loading");
  const audioToggle = document.getElementById("audioToggle");
  const backgroundAudio = document.getElementById("backgroundAudio");
  const audioIcon = document.getElementById("audioIcon");

  // Output containers
  const featuresOutput = document.getElementById("features-output");
  const makeupOutput = document.getElementById("makeup-output");
  const skincareOutput = document.getElementById("skincare-output");
  const tipsOutput = document.getElementById("tips-output");

  let currentStream;
  let videoDevices = [];
  let currentDeviceIndex = 0;
  let currentFacingMode = "user";
  let isAudioPlaying = false;

  // Initialize luxury theme
  initializeLuxuryTheme();

  function initializeLuxuryTheme() {
    // Add page load animation
    document.body.style.opacity = "0";
    setTimeout(() => {
      document.body.style.transition = "opacity 0.8s ease";
      document.body.style.opacity = "1";
    }, 100);

    // Add intersection observer for fade-in animations
    const observerOptions = {
      threshold: 0.1,
      rootMargin: "0px 0px -50px 0px",
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("fade-in");
        }
      });
    }, observerOptions);

    // Observe all cards for animation
    document
      .querySelectorAll(".feature-card, .product-card, .tip-card")
      .forEach((card) => {
        observer.observe(card);
      });

    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener("click", function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute("href"));
        if (target) {
          target.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }
      });
    });
  }

  function stopCurrentStream() {
    if (currentStream) {
      currentStream.getTracks().forEach((track) => {
        track.stop();
      });
    }
  }

  function getCameraDevices() {
    navigator.mediaDevices.enumerateDevices().then((devices) => {
      videoDevices = devices.filter((device) => device.kind === "videoinput");
      if (videoDevices.length > 1) {
        switchCameraButton.style.display = "block";
      }
    });
  }

  function startCamera() {
    stopCurrentStream();
    const constraints = {
      video: {
        facingMode: currentFacingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    };

    navigator.mediaDevices
      .getUserMedia(constraints)
      .then((stream) => {
        video.srcObject = stream;
        currentStream = stream;

        // Add elegant camera overlay animation
        video.addEventListener("loadedmetadata", function () {
          const overlay = document.querySelector(".camera-overlay");
          if (overlay) {
            overlay.style.opacity = "1";
            overlay.style.transform = "scale(1)";
          }

          // Apply horizontal flip to video display to fix mirror effect
          video.style.transform = "scaleX(-1)";
        });

        if (videoDevices.length === 0) {
          getCameraDevices();
        }
      })
      .catch((err) => {
        console.error("Error accessing camera: ", err);
        showError(
          "Camera access denied. Please allow camera permissions and refresh the page."
        );
      });
  }

  // Start with the default camera
  startCamera();

  switchCameraButton.addEventListener("click", () => {
    if (videoDevices.length > 1) {
      currentDeviceIndex = (currentDeviceIndex + 1) % videoDevices.length;
      startCamera(videoDevices[currentDeviceIndex].deviceId);
    } else {
      // Fallback to facing mode switch
      currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
      startCamera();
    }
  });

  // Capture and analyze
  captureButton.addEventListener("click", async () => {
    if (!currentStream) {
      showError(
        "Camera not available. Please allow camera access and try again."
      );
      return;
    }

    showLoading();

    try {
      // Capture frame with horizontal flip to fix mirror effect
      const context = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Apply horizontal flip transformation to fix mirror effect
      context.scale(-1, 1);
      context.translate(-canvas.width, 0);
      context.drawImage(video, 0, 0);

      // Convert to blob
      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.8)
      );

      // Create form data
      const formData = new FormData();
      formData.append("image", blob, "capture.jpg");

      console.log("Sending analysis request..."); // Debug log

      // Send to server
      const response = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });

      console.log("Response status:", response.status); // Debug log

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Server error response:", errorText);

        let errorMessage = "Analysis failed. Please try again.";
        try {
          const errorData = JSON.parse(errorText);
          if (errorData.error) {
            errorMessage = errorData.error;
          }
        } catch (e) {
          console.error("Could not parse error response:", e);
        }

        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log("Analysis completed successfully:", result); // Debug log

      // Check if result has error
      if (result.error) {
        throw new Error(result.error);
      }

      // Hide loading and show results with elegant animation
      hideLoading();
      displayResults(result);
    } catch (error) {
      console.error("Analysis error:", error);
      hideLoading();

      // Provide more specific error messages
      let errorMessage = error.message;
      if (error.message.includes("Failed to fetch")) {
        errorMessage =
          "Connection failed. Please check your internet connection and try again.";
      } else if (error.message.includes("No face detected")) {
        errorMessage =
          "No face detected. Please position your face clearly in the camera and try again.";
      } else if (error.message.includes("Invalid image")) {
        errorMessage =
          "Image processing failed. Please try again with a clearer image.";
      } else if (error.message.includes("Analysis failed")) {
        errorMessage =
          "Analysis failed. Please try again or contact support if the issue persists.";
      }

      showError(errorMessage);
    }
  });

  // Debug analysis
  debugButton.addEventListener("click", async () => {
    if (!currentStream) {
      showError("Camera not available");
      return;
    }

    showLoading();

    try {
      // Capture frame with horizontal flip to fix mirror effect
      const context = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Apply horizontal flip transformation to fix mirror effect
      context.scale(-1, 1);
      context.translate(-canvas.width, 0);
      context.drawImage(video, 0, 0);

      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.8)
      );
      const formData = new FormData();
      formData.append("image", blob, "debug.jpg");

      const response = await fetch("/debug", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Debug failed");
      }

      const debugData = await response.json();
      hideLoading();
      displayDebugResults(debugData);
    } catch (error) {
      console.error("Debug error: ", error);
      hideLoading();
      showError("Debug failed. Please try again.");
    }
  });

  // Audio Control with enhanced UX
  audioToggle.addEventListener("click", function () {
    if (isAudioPlaying) {
      backgroundAudio.pause();
      audioIcon.className = "fas fa-volume-mute";
      isAudioPlaying = false;

      // Add elegant mute animation
      audioToggle.style.transform = "scale(0.9)";
      setTimeout(() => {
        audioToggle.style.transform = "scale(1)";
      }, 150);
    } else {
      backgroundAudio
        .play()
        .catch((e) => console.log("Audio play failed: ", e));
      audioIcon.className = "fas fa-volume-up";
      isAudioPlaying = true;

      // Add elegant unmute animation
      audioToggle.style.transform = "scale(1.1)";
      setTimeout(() => {
        audioToggle.style.transform = "scale(1)";
      }, 150);
    }
  });

  // Enhanced Tab System with smooth transitions
  document.querySelectorAll(".tab-btn").forEach((button) => {
    button.addEventListener("click", function () {
      const targetTab = this.getAttribute("data-tab");

      // Remove active class from all tabs and panes
      document
        .querySelectorAll(".tab-btn")
        .forEach((btn) => btn.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((pane) => {
        pane.classList.remove("active");
        pane.style.opacity = "0";
        pane.style.transform = "translateY(20px)";
      });

      // Add active class to clicked tab and corresponding pane
      this.classList.add("active");
      const targetPane = document.getElementById(targetTab);
      targetPane.classList.add("active");

      // Smooth transition for active pane
      setTimeout(() => {
        targetPane.style.opacity = "1";
        targetPane.style.transform = "translateY(0)";
      }, 50);
    });
  });

  function showLoading() {
    loadingDiv.style.display = "block";
    resultsDiv.style.display = "none";

    // Add elegant fade-in animation
    setTimeout(() => {
      loadingDiv.style.opacity = "1";
      loadingDiv.style.transform = "scale(1)";
    }, 10);
  }

  function hideLoading() {
    loadingDiv.style.opacity = "0";
    loadingDiv.style.transform = "scale(0.95)";

    setTimeout(() => {
      loadingDiv.style.display = "none";
    }, 300);
  }

  function showError(message) {
    // Create elegant error notification
    const errorDiv = document.createElement("div");
    errorDiv.className = "error-notification";
    errorDiv.innerHTML = `
        <div class="error-content">
            <i class="fas fa-exclamation-triangle"></i>
            <span>${message}</span>
        </div>
    `;

    // Add CSS for error notification if not already present
    if (!document.getElementById("error-notification-style")) {
      const style = document.createElement("style");
      style.id = "error-notification-style";
      style.textContent = `
        .error-notification {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            color: white;
            padding: 15px 25px;
            border-radius: 25px;
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
            z-index: 1000;
            animation: slideDown 0.3s ease;
        }
        
        .error-content {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }
        
        @keyframes slideDown {
            from { transform: translateX(-50%) translateY(-100%); opacity: 0; }
            to { transform: translateX(-50%) translateY(0); opacity: 1; }
        }
        
        @keyframes slideUp {
            from { transform: translateX(-50%) translateY(0); opacity: 1; }
            to { transform: translateX(-50%) translateY(-100%); opacity: 0; }
        }
      `;
      document.head.appendChild(style);
    }

    document.body.appendChild(errorDiv);

    // Remove after 5 seconds with elegant animation
    setTimeout(() => {
      errorDiv.style.animation = "slideUp 0.3s ease";
      setTimeout(() => {
        if (document.body.contains(errorDiv)) {
          document.body.removeChild(errorDiv);
        }
      }, 300);
    }, 5000);
  }

  function displayResults(data) {
    console.log("Displaying results:", data);

    // Display features
    if (data.features) {
      const featuresOutput = document.getElementById("features-output");
      featuresOutput.innerHTML = "";

      Object.entries(data.features).forEach(([key, feature]) => {
        const featureCard = document.createElement("div");
        featureCard.className = "feature-card";
        featureCard.innerHTML = `
                <div class="feature-icon">
                    <i class="fas ${getFeatureIcon(key)}"></i>
                </div>
                <div class="feature-content">
                    <h4>${feature.value}</h4>
                    <p>${feature.description}</p>
                </div>
            `;
        featuresOutput.appendChild(featureCard);
      });
    }

    // Display makeup recommendations
    if (data.makeup_recommendations && data.makeup_recommendations.length > 0) {
      const makeupOutput = document.getElementById("makeup-output");
      makeupOutput.innerHTML = "";

      data.makeup_recommendations.forEach((product) => {
        const productCard = document.createElement("div");
        productCard.className = "product-card";
        productCard.innerHTML = `
                <h6>${product.name}</h6>
                <p class="product-type">${product.type || "Makeup"}</p>
                <p class="product-price">${product.price}</p>
                <p class="product-brand">${product.brand}</p>
            `;
        makeupOutput.appendChild(productCard);
      });
    }

    // Display skincare recommendations
    if (
      data.skincare_recommendations &&
      data.skincare_recommendations.length > 0
    ) {
      const skincareOutput = document.getElementById("skincare-output");
      skincareOutput.innerHTML = "";

      data.skincare_recommendations.forEach((product) => {
        const productCard = document.createElement("div");
        productCard.className = "product-card";
        productCard.innerHTML = `
                <h6>${product.name}</h6>
                <p class="product-type">${product.type || "Skincare"}</p>
                <p class="product-price">${product.price}</p>
                <p class="product-brand">${product.brand}</p>
            `;
        skincareOutput.appendChild(productCard);
      });
    }

    // Display beauty tips
    if (data.tips && data.tips.length > 0) {
      const tipsOutput = document.getElementById("tips-output");
      tipsOutput.innerHTML = "";

      data.tips.forEach((tip) => {
        const tipCard = document.createElement("div");
        tipCard.className = "tip-card";
        tipCard.innerHTML = `
                <div class="tip-icon">
                    <i class="fas fa-lightbulb"></i>
                </div>
                <div class="tip-content">
                    <p>${tip}</p>
                </div>
            `;
        tipsOutput.appendChild(tipCard);
      });
    }

    // Display comprehensive beauty analysis
    if (data.beauty_analysis) {
      displayBeautyAnalysis(data.beauty_analysis);
    }

    // Show results
    document.getElementById("results").style.display = "block";
    document.getElementById("results").scrollIntoView({ behavior: "smooth" });
  }

  function displayBeautyAnalysis(beautyAnalysis) {
    // Lip Colors Analysis
    if (beautyAnalysis.lip_colors) {
      const lipAnalysis = beautyAnalysis.lip_colors;

      // Display best colors
      const bestLipColors = document.getElementById("best-lip-colors");
      bestLipColors.innerHTML = "";
      lipAnalysis.best_colors.forEach((color) => {
        const colorTag = document.createElement("span");
        colorTag.className = "color-tag";
        colorTag.textContent = color;
        bestLipColors.appendChild(colorTag);
      });

      // Display avoid colors
      const avoidLipColors = document.getElementById("avoid-lip-colors");
      avoidLipColors.innerHTML = "";
      lipAnalysis.avoid_colors.forEach((color) => {
        const colorTag = document.createElement("span");
        colorTag.className = "color-tag";
        colorTag.textContent = color;
        avoidLipColors.appendChild(colorTag);
      });

      // Display technique
      document.getElementById("lip-technique").textContent =
        lipAnalysis.technique;

      // Display products
      const lipProducts = document.getElementById("lip-products");
      lipProducts.innerHTML = "";
      lipAnalysis.products.forEach((product) => {
        const productCard = createProductCard(product);
        lipProducts.appendChild(productCard);
      });
    }

    // Blush Analysis
    if (beautyAnalysis.blush_shades) {
      const blushAnalysis = beautyAnalysis.blush_shades;

      // Display best shades
      const bestBlushShades = document.getElementById("best-blush-shades");
      bestBlushShades.innerHTML = "";
      blushAnalysis.best_shades.forEach((shade) => {
        const colorTag = document.createElement("span");
        colorTag.className = "color-tag";
        colorTag.textContent = shade;
        bestBlushShades.appendChild(colorTag);
      });

      // Display avoid shades
      const avoidBlushShades = document.getElementById("avoid-blush-shades");
      avoidBlushShades.innerHTML = "";
      blushAnalysis.avoid_shades.forEach((shade) => {
        const colorTag = document.createElement("span");
        colorTag.className = "color-tag";
        colorTag.textContent = shade;
        avoidBlushShades.appendChild(colorTag);
      });

      // Display application
      document.getElementById("blush-application").textContent =
        blushAnalysis.application;

      // Display products
      const blushProducts = document.getElementById("blush-products");
      blushProducts.innerHTML = "";
      blushAnalysis.products.forEach((product) => {
        const productCard = createProductCard(product);
        blushProducts.appendChild(productCard);
      });
    }

    // Eye Makeup Analysis
    if (beautyAnalysis.eye_makeup) {
      const eyeAnalysis = beautyAnalysis.eye_makeup;

      document.getElementById("eyeliner-style").textContent =
        eyeAnalysis.eyeliner_style;
      document.getElementById("eyeshadow-technique").textContent =
        eyeAnalysis.eyeshadow_technique;
      document.getElementById("mascara-technique").textContent =
        eyeAnalysis.mascara_technique;
      document.getElementById("eye-avoid").textContent = eyeAnalysis.avoid;

      // Display technique steps
      const eyeSteps = document.getElementById("eye-technique-steps");
      eyeSteps.innerHTML = "";
      eyeAnalysis.technique_steps.forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        eyeSteps.appendChild(li);
      });

      // Display products
      const eyeProducts = document.getElementById("eye-products");
      eyeProducts.innerHTML = "";
      eyeAnalysis.products.forEach((product) => {
        const productCard = createProductCard(product);
        eyeProducts.appendChild(productCard);
      });
    }

    // Nose Contouring Analysis
    if (beautyAnalysis.nose_contouring) {
      const noseAnalysis = beautyAnalysis.nose_contouring;

      // Display contouring status
      const noseStatus = document.getElementById("nose-contouring-status");
      noseStatus.className = `contouring-status ${
        noseAnalysis.needs_contouring ? "needed" : "not-needed"
      }`;
      noseStatus.innerHTML = `
            <h5>${
              noseAnalysis.needs_contouring
                ? "Contouring Recommended"
                : "No Contouring Needed"
            }</h5>
            <p>${noseAnalysis.technique}</p>
        `;

      // Display contouring areas
      const noseContouringAreas = document.getElementById(
        "nose-contouring-areas"
      );
      noseContouringAreas.innerHTML = "";
      noseAnalysis.contouring_areas.forEach((area) => {
        const li = document.createElement("li");
        li.textContent = area;
        noseContouringAreas.appendChild(li);
      });

      // Display highlighting areas
      const noseHighlightAreas = document.getElementById(
        "nose-highlight-areas"
      );
      noseHighlightAreas.innerHTML = "";
      noseAnalysis.highlight_areas.forEach((area) => {
        const li = document.createElement("li");
        li.textContent = area;
        noseHighlightAreas.appendChild(li);
      });

      // Display steps
      const noseSteps = document.getElementById("nose-steps");
      noseSteps.innerHTML = "";
      noseAnalysis.steps.forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        noseSteps.appendChild(li);
      });

      // Display products
      const noseProducts = document.getElementById("nose-products");
      noseProducts.innerHTML = "";
      noseAnalysis.products.forEach((product) => {
        const productCard = createProductCard(product);
        noseProducts.appendChild(productCard);
      });
    }

    // Face Contouring Analysis
    if (beautyAnalysis.face_contouring) {
      const faceAnalysis = beautyAnalysis.face_contouring;

      // Display contouring status
      const faceStatus = document.getElementById("face-contouring-status");
      faceStatus.className = `contouring-status ${
        faceAnalysis.needs_contouring ? "needed" : "not-needed"
      }`;
      faceStatus.innerHTML = `
            <h5>${
              faceAnalysis.needs_contouring
                ? "Contouring Recommended"
                : "No Contouring Needed"
            }</h5>
            <p>${faceAnalysis.technique}</p>
        `;

      // Display contouring areas
      const faceContouringAreas = document.getElementById(
        "face-contouring-areas"
      );
      faceContouringAreas.innerHTML = "";
      faceAnalysis.contouring_areas.forEach((area) => {
        const li = document.createElement("li");
        li.textContent = area;
        faceContouringAreas.appendChild(li);
      });

      // Display highlighting areas
      const faceHighlightAreas = document.getElementById(
        "face-highlight-areas"
      );
      faceHighlightAreas.innerHTML = "";
      faceAnalysis.highlight_areas.forEach((area) => {
        const li = document.createElement("li");
        li.textContent = area;
        faceHighlightAreas.appendChild(li);
      });

      // Display steps
      const faceSteps = document.getElementById("face-steps");
      faceSteps.innerHTML = "";
      faceAnalysis.steps.forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        faceSteps.appendChild(li);
      });

      // Display products
      const faceProducts = document.getElementById("face-products");
      faceProducts.innerHTML = "";
      faceAnalysis.products.forEach((product) => {
        const productCard = createProductCard(product);
        faceProducts.appendChild(productCard);
      });
    }
  }

  function createProductCard(product) {
    const productCard = document.createElement("div");
    productCard.className = "product-card";
    productCard.innerHTML = `
        <h6>${product.name}</h6>
        <p class="product-type">${product.type || "Product"}</p>
        <p class="product-price">${product.price}</p>
        <p class="product-brand">${product.brand}</p>
    `;
    return productCard;
  }

  function displayDebugResults(debugData) {
    resultsDiv.style.display = "block";
    resultsDiv.style.opacity = "0";
    resultsDiv.style.transform = "translateY(20px)";

    // Create a more organized debug display
    let debugHTML = '<div class="debug-panel">';
    debugHTML += '<h4><i class="fas fa-bug"></i> Debug Information</h4>';

    // Image info
    if (debugData.image_size) {
      debugHTML += `<div class="debug-section">
        <h5><i class="fas fa-image"></i> Image Information</h5>
        <p><strong>Size:</strong> ${debugData.image_size}</p>
        <p><strong>Landmarks Detected:</strong> ${debugData.landmarks_detected}</p>
      </div>`;
    }

    // Face shape analysis
    if (debugData.face_shape_analysis) {
      const face = debugData.face_shape_analysis;
      debugHTML += `<div class="debug-section">
        <h5><i class="fas fa-user"></i> Face Shape Analysis</h5>
        <div class="debug-grid">
          <div><strong>Face Width:</strong> ${face.face_width}</div>
          <div><strong>Face Height:</strong> ${face.face_height}</div>
          <div><strong>Height/Width Ratio:</strong> ${face.height_width_ratio}</div>
          <div><strong>Cheek/Jaw Ratio:</strong> ${face.cheek_jaw_ratio}</div>
        </div>
      </div>`;
    }

    // Skin tone analysis
    if (debugData.skin_tone_analysis) {
      const skin = debugData.skin_tone_analysis;
      debugHTML += `<div class="debug-section">
        <h5><i class="fas fa-palette"></i> Skin Tone Analysis</h5>
        <div class="debug-grid">
          <div><strong>Color Samples:</strong> ${skin.color_samples_count}</div>
          <div><strong>Average RGB:</strong> [${skin.average_rgb.join(
            ", "
          )}]</div>
          <div><strong>Red Value:</strong> ${skin.red_value}</div>
          <div><strong>Red-Green Diff:</strong> ${skin.red_green_diff}</div>
          <div><strong>Red-Blue Diff:</strong> ${skin.red_blue_diff}</div>
        </div>
      </div>`;
    }

    // Eye analysis
    if (debugData.eye_analysis) {
      const eye = debugData.eye_analysis;
      debugHTML += `<div class="debug-section">
        <h5><i class="fas fa-eye"></i> Eye Analysis</h5>
        <div class="debug-grid">
          <div><strong>Left Eye Width:</strong> ${eye.left_eye_width}</div>
          <div><strong>Left Eye Height:</strong> ${eye.left_eye_height}</div>
          <div><strong>Right Eye Width:</strong> ${eye.right_eye_width}</div>
          <div><strong>Right Eye Height:</strong> ${eye.right_eye_height}</div>
          <div><strong>Average Eye Width:</strong> ${eye.avg_eye_width}</div>
          <div><strong>Average Eye Height:</strong> ${eye.avg_eye_height}</div>
          <div><strong>Eye Ratio:</strong> ${eye.eye_ratio}</div>
        </div>
      </div>`;
    }

    // Lip analysis
    if (debugData.lip_analysis) {
      const lip = debugData.lip_analysis;
      debugHTML += `<div class="debug-section">
        <h5><i class="fas fa-kiss-wink-heart"></i> Lip Analysis</h5>
        <div class="debug-grid">
          <div><strong>Lip Width:</strong> ${lip.lip_width}</div>
          <div><strong>Lip Height:</strong> ${lip.lip_height}</div>
          <div><strong>Lip Ratio:</strong> ${lip.lip_ratio}</div>
          <div><strong>Upper Lip Height:</strong> ${lip.upper_lip_height}</div>
          <div><strong>Lower Lip Height:</strong> ${lip.lower_lip_height}</div>
          <div><strong>Total Lip Height:</strong> ${lip.total_lip_height}</div>
          <div><strong>Height/Width Ratio:</strong> ${lip.height_width_ratio}</div>
        </div>
      </div>`;
    }

    // Nose analysis
    if (debugData.nose_analysis) {
      const nose = debugData.nose_analysis;
      if (nose.error) {
        debugHTML += `<div class="debug-section">
          <h5><i class="fas fa-smile"></i> Nose Analysis</h5>
          <p class="debug-error">${nose.error}</p>
        </div>`;
      } else {
        debugHTML += `<div class="debug-section">
          <h5><i class="fas fa-smile"></i> Nose Analysis</h5>
          <div class="debug-grid">
            <div><strong>Nose Length:</strong> ${nose.nose_length}</div>
            <div><strong>Nose Width:</strong> ${nose.nose_width}</div>
            <div><strong>Nose Ratio:</strong> ${nose.nose_ratio}</div>
            <div><strong>Bridge Width:</strong> ${nose.bridge_width}</div>
            <div><strong>Tip Width:</strong> ${nose.tip_width}</div>
            <div><strong>Bridge/Tip Ratio:</strong> ${nose.bridge_tip_ratio}</div>
            <div><strong>Base Ratio:</strong> ${nose.base_ratio}</div>
            <div><strong>Bridge Curvature:</strong> ${nose.bridge_curvature}</div>
            <div><strong>Tip Angle:</strong> ${nose.tip_angle}</div>
            <div><strong>Asymmetry:</strong> ${nose.asymmetry}</div>
          </div>
          <div class="debug-subsection">
            <h6>Landmark Coordinates:</h6>
            <div class="debug-grid">
              <div><strong>Bridge Top:</strong> (${nose.measurements.bridge_top[0]}, ${nose.measurements.bridge_top[1]})</div>
              <div><strong>Bridge Mid:</strong> (${nose.measurements.bridge_mid[0]}, ${nose.measurements.bridge_mid[1]})</div>
              <div><strong>Bridge Lower:</strong> (${nose.measurements.bridge_lower[0]}, ${nose.measurements.bridge_lower[1]})</div>
              <div><strong>Nose Tip:</strong> (${nose.measurements.nose_tip[0]}, ${nose.measurements.nose_tip[1]})</div>
              <div><strong>Nose Bottom:</strong> (${nose.measurements.nose_bottom[0]}, ${nose.measurements.nose_bottom[1]})</div>
              <div><strong>Nose Left:</strong> (${nose.measurements.nose_left[0]}, ${nose.measurements.nose_left[1]})</div>
              <div><strong>Nose Right:</strong> (${nose.measurements.nose_right[0]}, ${nose.measurements.nose_right[1]})</div>
            </div>
          </div>
        </div>`;
      }
    }

    // Raw data (collapsible)
    debugHTML += `<div class="debug-section">
      <h5><i class="fas fa-code"></i> Raw Data</h5>
      <details>
        <summary>Click to expand raw JSON data</summary>
        <pre class="debug-raw">${JSON.stringify(debugData, null, 2)}</pre>
      </details>
    </div>`;

    debugHTML += "</div>";

    featuresOutput.innerHTML = debugHTML;

    // Switch to features tab
    document
      .querySelectorAll(".tab-btn")
      .forEach((btn) => btn.classList.remove("active"));
    document
      .querySelectorAll(".tab-pane")
      .forEach((pane) => pane.classList.remove("active"));

    document.querySelector('[data-tab="features"]').classList.add("active");
    document.getElementById("features").classList.add("active");

    setTimeout(() => {
      resultsDiv.style.opacity = "1";
      resultsDiv.style.transform = "translateY(0)";
    }, 10);
  }

  function formatFeatureName(feature) {
    return feature
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  function getFeatureIcon(feature) {
    const iconMap = {
      face_shape: "fa-user",
      skin_tone: "fa-palette",
      skin_type: "fa-spa",
      eye_shape: "fa-eye",
      lip_shape: "fa-kiss-wink-heart",
      nose_shape: "fa-smile",
    };

    return iconMap[feature] || "fa-star";
  }
});
