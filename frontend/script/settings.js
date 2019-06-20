$.get("/api/settings", function(data) {
  let html = "";
  for (let dataset_name in data.DATASETS) {
    let dataset_values = data.DATASETS[dataset_name];
    html += `
      <div class="dataset" dataset="${dataset_name}">
        <h2>${dataset_name}</h2>
        <label>
          Name
          <input name="${dataset_name}_name" class="dataset_name" type="text"
              val="${dataset_name}">
        </label>
        <label>
          Database URL
          <input name="${dataset_name}_url" class="dataset_url" type="text"
              val="${dataset_values.DB_URL}">
        </label>
        <label>
          Metric Column
          <input name="${dataset_name}_metric" class="dataset_metric"
              type="text" val="${dataset_values.METRIC_COLUMN}">
        </label>
      </div>
      `;
  }
  $("#settings h1").after(html);
});

$("#submit").click(function () {
  let data = {};
  $(".dataset").each(function (i) {
    let name = $(this).find(".dataset_name").val();
    let url = $(this).find(".dataset_url").val();
    let metric = $(this).find(".dataset_metric").val();
    let dataset_id = $(this).attr("dataset");
    data[dataset_id] = {
      "NAME" : name,
      "DB_URL" : url,
      "METRIC_COLUMN" : metric
    };
  });
  $.post("/api/settings", {"DATASETS" : data});
})
