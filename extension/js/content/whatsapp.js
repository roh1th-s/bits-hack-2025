MutationObserver = window.MutationObserver || window.WebKitMutationObserver;

function addButton() {
  let a = document.getElementsByClassName(
    "x1n2onr6 xfo81ep x9f619 x78zum5 x6s0dn4 xh8yej3 x7j6532 x1pl83jw x1j6awrg x1te75w5 x162n7g1 x4eaejv xcock1l x1s928wv xl9llhp x1qj619r x1r9ni5o xvkby78 x889kno x1a8lsjc x1swvt13 x1pi30zi xu306ak x1h1zc6f"
  )[0];
  if (!a) {
    return false;
  }

  // Create button element
  const button = document.createElement("button");
  button.className = "whatsapp";
  button.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" height="12" width="15" viewBox="0 0 640 512"><!--!Font Awesome Free 6.7.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.--><path fill="#ffffff" d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512l388.6 0c1.8 0 3.5-.2 5.3-.5c-76.3-55.1-99.8-141-103.1-200.2c-16.1-4.8-33.1-7.3-50.7-7.3l-91.4 0zm308.8-78.3l-120 48C358 277.4 352 286.2 352 296c0 63.3 25.9 168.8 134.8 214.2c5.9 2.5 12.6 2.5 18.5 0C614.1 464.8 640 359.3 640 296c0-9.8-6-18.6-15.1-22.3l-120-48c-5.7-2.3-12.1-2.3-17.8 0zM591.4 312c-3.9 50.7-27.2 116.7-95.4 149.7l0-187.8L591.4 312z"/></svg>
    <span>Fence Check</span>
`;

  // Append button to body
  document.body.appendChild(button);

  // Create style element
  const style = document.createElement("style");
  style.textContent = `
        button.whatsapp {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 3px;
            padding: 8px 14px;
            font-size: 12px;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            background-color: #25D366;
            color: white;
            cursor: pointer;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        button:hover.whatsapp {
            background-color: #1EBE5D;
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.15);
            transform: scale(1.06);
        }
        button svg {
            width: 16px;
            margin-right: 4px;
            height: 16px;
            position: relative;
        }
    `;

  // Append styles to head
  document.head.appendChild(style);

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

  var observer = new MutationObserver((mutations) => {
    let newChatOpened = mutations.some((mutation) => {
      if (mutation.addedNodes.length) {
        return Array.from(mutation.addedNodes).some(
          (node) => node.id == "main" && node.className.includes("_ajx_")
        );
      }
      return false;
    });
    if (newChatOpened) {
      addButton();
    }
  });

  let b = document.getElementsByClassName(
    "x9f619 x1n2onr6 xyw6214 x5yr21d x6ikm8r x10wlt62 x17dzmu4 x1i1dayz x2ipvbc x1w8yi2h xyyilfv x1iyjqo2 xy80clv x26u7qi x1ux35ld"
  )[0];
  if (!b) {
    console.log("CANT FIND RIGHT PANEL");

    return;
  }
  observer.observe(b, {
    subtree: true,
    childList: true,
  });
}
var observer = new MutationObserver(callback);

observer.observe(document, {
  subtree: true,
  childList : true,
  });
