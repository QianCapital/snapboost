document.addEventListener("DOMContentLoaded", function () {
  var search = document.querySelector('.wy-side-nav-search input[type="text"]');
  if (search) {
    search.setAttribute("placeholder", "Search docs");
  }

  var year = String(new Date().getFullYear());
  var copyright = document.querySelector('[role="contentinfo"] p');
  if (copyright) {
    copyright.innerHTML = copyright.innerHTML.replace(/\b(?:19|20)\d{2}\b/, year);
  }

  placeSidebarBrand();
  initVersionSwitcher();
});

function placeSidebarBrand() {
  var brand = document.querySelector(".sb-sidebar-brand");
  var side = document.querySelector(".wy-nav-side");
  if (!brand || !side) {
    return;
  }
  side.appendChild(brand);
  brand.hidden = false;
}

function initVersionSwitcher() {
  var select = document.getElementById("sb-version-select");
  if (!select) {
    return;
  }

  select.addEventListener("change", function () {
    if (select.value) {
      window.location.href = select.value;
    }
  });

  var currentVersion = select.getAttribute("data-current") || "latest";

  fetch("/versions.json", { credentials: "same-origin" })
    .then(function (response) {
      if (!response.ok) {
        throw new Error("versions.json " + response.status);
      }
      return response.json();
    })
    .then(function (versions) {
      if (!Array.isArray(versions) || !versions.length) {
        return;
      }
      select.innerHTML = "";
      versions.forEach(function (entry) {
        var option = document.createElement("option");
        option.value = entry.url;
        option.textContent = entry.version;
        if (entry.version === currentVersion) {
          option.selected = true;
        }
        select.appendChild(option);
      });
    })
    .catch(function () {
      // Keep Sphinx-baked options for local builds / offline viewing.
    });
}
