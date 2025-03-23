MutationObserver = window.MutationObserver || window.WebKitMutationObserver;
function addButton() {
  let a = document.getElementsByClassName(
    "x6s0dn4 x78zum5 x1q0g3np xs83m0k xeuugli x1n2onr6"
  )[0];
  if (!a) {
    return false;
  }
  // Check if button already exists
  if (document.querySelector(".instagram.buttonmain")) {
    return true;
  }

  // Create button element
  const button = document.createElement("button");
  button.className = "instagram";
  button.classList.add("instagram", "buttonmain");
  button.innerHTML = `
<svg xmlns="http://www.w3.org/2000/svg" height="12" width="15" viewBox="0 0 640 512">
<path fill="#ffffff" d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512l388.6 0c1.8 0 3.5-.2 5.3-.5c-76.3-55.1-99.8-141-103.1-200.2c-16.1-4.8-33.1-7.3-50.7-7.3l-91.4 0zm308.8-78.3l-120 48C358 277.4 352 286.2 352 296c0 63.3 25.9 168.8 134.8 214.2c5.9 2.5 12.6 2.5 18.5 0C614.1 464.8 640 359.3 640 296c0-9.8-6-18.6-15.1-22.3l-120-48c-5.7-2.3-12.1-2.3-17.8 0zM591.4 312c-3.9 50.7-27.2 116.7-95.4 149.7l0-187.8L591.4 312z"/></svg>
<span>Fence Check</span>
    `;

  // Apply styles dynamically
  Object.assign(button.style, {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "3px",
    padding: "8px 14px",
    fontSize: "12px",
    fontWeight: "600",
    border: "none",
    borderRadius: "8px",
    height: "32px",
    background:
      "linear-gradient(45deg, #f09433, #e6683c, #dc2743, #cc2366, #bc1888)",
    color: "white",
    cursor: "pointer",
    transition: "transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out",
  });

  button.addEventListener("mouseover", () => {
    button.style.transform = "scale(1.06)";
    button.style.boxShadow = "0px 4px 10px rgba(0, 0, 0, 0.2)";
  });

  button.addEventListener("mouseout", () => {
    button.style.transform = "scale(1)";
    button.style.boxShadow = "none";
  });

  // Add click event to load stagedpopup.js
  button.addEventListener("click", () => {
    setupPopupForButtons();
  });

  a.appendChild(button);
  
  return true;
}

function callback(mutations, observer) {
  let addedButton = addButton();
  if (!addedButton) {
    console.log("Button not added");
    return;
  }

  // Stop observing to avoid infinite loops
  observer.disconnect();

  var observer = new MutationObserver((mutations) => {
    let newProfileOpened = mutations.some((mutation) => {
      if (mutation.addedNodes.length) {
        return Array.from(mutation.addedNodes).some((node) => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            return (
              (node.className &&
                node.className.includes("x78zum5 xdt5ytf x1iyjqo2 xg6iff7")) ||
              node.querySelector(".x78zum5.xdt5ytf.x1iyjqo2.xg6iff7")
            );
          }
          return false;
        });
      }
      return false;
    });

    if (newProfileOpened) {
      console.log("New profile opened");
      addButton();
    }
  });

  observer.observe(document, {
    subtree: true,
    childList: true,
  });
}

var observer = new MutationObserver(callback);
observer.observe(document, {
  subtree: true,
  childList: true,
});
