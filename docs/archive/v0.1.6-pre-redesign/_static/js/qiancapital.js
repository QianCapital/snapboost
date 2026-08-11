document.addEventListener("DOMContentLoaded", function () {
  var search = document.querySelector('.wy-side-nav-search input[type="text"]');
  if (search && !search.getAttribute("placeholder")) {
    search.setAttribute("placeholder", "Search docs…");
  }

  var year = String(new Date().getFullYear());
  var copyright = document.querySelector('[role="contentinfo"] p');
  if (copyright) {
    copyright.innerHTML = copyright.innerHTML.replace(/\b(?:19|20)\d{2}\b/, year);
  }
});
