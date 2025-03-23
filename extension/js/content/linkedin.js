MutationObserver = window.MutationObserver || window.WebKitMutationObserver;

function addButton() {
  let a = document.getElementsByClassName(
    "IdHDOXrWZTHgjeHGKHPGcqeJRsFoXHCMlxbI"
  )[0];
  if (!a) {
    return false;
  }
  // Create the button element
      
  const button = document.createElement("button");
  button.classList.add("linkedin");
  button.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" height="12" width="15" viewBox="0 0 640 512"><!--!Font Awesome Free 6.7.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.--><path fill="#ffffff" d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512l388.6 0c1.8 0 3.5-.2 5.3-.5c-76.3-55.1-99.8-141-103.1-200.2c-16.1-4.8-33.1-7.3-50.7-7.3l-91.4 0zm308.8-78.3l-120 48C358 277.4 352 286.2 352 296c0 63.3 25.9 168.8 134.8 214.2c5.9 2.5 12.6 2.5 18.5 0C614.1 464.8 640 359.3 640 296c0-9.8-6-18.6-15.1-22.3l-120-48c-5.7-2.3-12.1-2.3-17.8 0zM591.4 312c-3.9 50.7-27.2 116.7-95.4 149.7l0-187.8L591.4 312z"/></svg>
    <span>Fence Check</span>
`;

  // Add styles dynamically
  const style = document.createElement("style");
  style.textContent = `
    .linkedin {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 3px;
        height: 32px;
        padding: 6px 16px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        border-radius: 1.6rem;
        background-color:rgb(10, 102, 194);
        color: white;
        cursor: pointer;
        transition: background-color 0.2s ease-in-out, transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .linkedin:hover {
        background-color: rgb(9, 90, 171);
        box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.15);
        transform: scale(1.06);
    }
    .linkedin svg {
        width: 16px;
        height: 16px;
        margin-right: 4px;
        position: relative;
    }
`;

  // Append the styles and button to the document
  document.head.appendChild(style);
  document.body.appendChild(button);
  a.appendChild(button);
  return true;
}

function callback(mutations, observer) {
  let addedButton = addButton();
  if (!addedButton) {
    console.log("Button not added");

    return;
  }

  //delete whole doument observer
  observer.disconnect();

  console.log("issue");
  
  var observer = new MutationObserver((mutations) => {
    let newProfileOpened = mutations.some((mutation) => {
      if (mutation.addedNodes.length) {
        return Array.from(mutation.addedNodes).some(
          (node) =>  (node.className && node.className.includes("IdHDOXrWZTHgjeHGKHPGcqeJRsFoXHCMlxbI")) || node.querySelector(".IdHDOXrWZTHgjeHGKHPGcqeJRsFoXHCMlxbI")
        );
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
