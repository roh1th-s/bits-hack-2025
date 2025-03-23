document.addEventListener("DOMContentLoaded", function () {
  const toggle1 = document.getElementById("toggle1");

  chrome.storage.local.get(["toggleStates"], function (result) {
    const savedStates = result.toggleStates || {};

    if (savedStates.toggle1 !== undefined)
      toggle1.checked = savedStates.toggle1;
  });

  toggle1.addEventListener("change", saveToggleStates);

  // Function to save all toggle states
  function saveToggleStates() {
    const toggleStates = {
      toggle1: toggle1.checked,
    };

    // Save to local storage
    chrome.storage.local.set({ toggleStates: toggleStates });
  }
});
