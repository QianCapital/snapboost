document.addEventListener("DOMContentLoaded", function () {
  var search = document.querySelector('.wy-side-nav-search input[type="text"]');
  if (search && !search.getAttribute("placeholder")) {
    search.setAttribute("placeholder", "Search docs…");
  }
});
