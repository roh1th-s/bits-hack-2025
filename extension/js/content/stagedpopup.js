// Staged Popup Script for Buttons
// This script creates a popup with four timed stages when a button is clicked

// Main function to set up popup functionality
function setupPopupForButtons() {
  console.log("awdawd");

  // Create the popup elements
  const popup = document.createElement("div");
  const popupContent = document.createElement("div");

  // Add logo container
  const logoContainer = document.createElement("div");
  const logo = document.createElement("img");

  const messageContainer = document.createElement("div");
  const message = document.createElement("div");
  const loadingIcon = document.createElement("div");

  // Set up classes for styling
  popup.className = "multi-stage-popup";
  popupContent.className = "popup-content";
  logoContainer.className = "logo-container";
  messageContainer.className = "message-container";
  message.className = "popup-message";
  loadingIcon.className = "loading-circle";

  // Set the logo source - replace this with your actual data URI
  logo.src =
    "data:image/png;base64,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";
  logo.alt = "Logo";
  logo.className = "popup-logo";

  // Assemble popup structure
  logoContainer.appendChild(logo);
  messageContainer.appendChild(message);
  messageContainer.appendChild(loadingIcon);

  popupContent.appendChild(logoContainer);
  popupContent.appendChild(messageContainer);
  popup.appendChild(popupContent);

  // Add styles to the document
  const styles = document.createElement("style");
  styles.textContent = `
      .multi-stage-popup {
        display: none;
        position: fixed;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.7);
        z-index: 9999;
        justify-content: center;
        align-items: center;
      }
      
      .popup-content {
        background-color: black;
        padding: 40px;
        border-radius: 12px;
        width: 600px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
      }
      
      .logo-container {
        margin-bottom: 30px;
        text-align: center;
      }
      
      .popup-logo {
        max-width: 150px;
        max-height: 150px;
      }
      
      .message-container {
        width: 100%;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
      }
      
      .popup-message {
        font-size: 36px;
        font-weight: 900;
        color: white;
        text-align: left;
        flex: 1;
        margin-right: 40px;
      }
      
      .loading-circle {
        width: 100px;
        height: 100px;
        border: 8px solid rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        border-top: 8px solid #3498db;
        animation: spin 1s linear infinite;
        flex-shrink: 0;
      }
      
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      
      /* Separate animations for fading in and out */
      @keyframes fadeInMessage {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
      }
      
      @keyframes fadeOutMessage {
        0% { opacity: 1; transform: translateY(0); }
        100% { opacity: 0; transform: translateY(-10px); }
      }
      
      .message-transition-in {
        animation: fadeInMessage 1s ease-in-out forwards;
      }
      
      .message-transition-out {
        animation: fadeOutMessage 1s ease-in-out forwards;
      }
      
      /* Mobile responsiveness */
      @media (max-width: 700px) {
        .popup-content {
          width: 90%;
          padding: 30px;
        }
        
        .message-container {
          flex-direction: column;
        }
        
        .popup-message {
          font-size: 28px;
          margin-right: 0;
          margin-bottom: 30px;
          text-align: center;
        }
        
        .popup-logo {
          max-width: 100px;
          max-height: 100px;
        }
      }
    `;

  // Add elements to the document
  document.head.appendChild(styles);
  document.body.appendChild(popup);

  // Function to show popup and run the timed stages
  function showTimedPopup() {
    // Display the popup
    popup.style.display = "flex";

    // Stage 1: Fetching data (0-7 seconds)
    message.textContent = "Fetching data";
    message.classList.add("message-transition-in");
    loadingIcon.style.borderTopColor = "purple";

    // Stage 2: Scanning suspiciously (7-14 seconds)
    setTimeout(() => {
      // Fade out current message
      message.classList.remove("message-transition-in");
      message.classList.add("message-transition-out");

      // After fade out completes, change message and fade in
      setTimeout(() => {
        message.textContent = "Scanning for suspicion";
        loadingIcon.style.borderTopColor = "green";
        message.classList.remove("message-transition-out");
        message.classList.add("message-transition-in");
      }, 1000);
    }, 7000);

    // Stage 3: Making deduction (14-21 seconds)
    setTimeout(() => {
      // Fade out current message
      message.classList.remove("message-transition-in");
      message.classList.add("message-transition-out");

      // After fade out completes, change message and fade in
      setTimeout(() => {
        message.textContent = "Making deductions";
        loadingIcon.style.borderTopColor = "blue";
        message.classList.remove("message-transition-out");
        message.classList.add("message-transition-in");
      }, 1000);
    }, 14000);

    // Stage 4: Setting up fences (21-26 seconds)
    setTimeout(() => {
      // Fade out current message
      message.classList.remove("message-transition-in");
      message.classList.add("message-transition-out");

      // After fade out completes, change message and fade in
      setTimeout(() => {
        message.textContent = "Setting up fences";
        loadingIcon.style.borderTopColor = "grey";
        message.classList.remove("message-transition-out");
        message.classList.add("message-transition-in");
      }, 1000);
    }, 21000);

    // Fade out before closing
    setTimeout(() => {
      message.classList.remove("message-transition-in");
      message.classList.add("message-transition-out");

      // Close popup after fade out
      setTimeout(() => {
        popup.style.display = "none";
        message.classList.remove("message-transition-out");
      }, 1000);
    }, 26000);
  }

  showTimedPopup();
}
