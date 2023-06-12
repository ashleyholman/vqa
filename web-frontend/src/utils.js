function convertToLocalTimestamp(utcTimestamp) {
  const utcDate = new Date(utcTimestamp + "Z");
  const localTimestamp = utcDate.getFullYear() + "-" +
                         ("0" + (utcDate.getMonth()+1)).slice(-2) + "-" +
                         ("0" + utcDate.getDate()).slice(-2) + " " +
                         ("0" + utcDate.getHours()).slice(-2) + ":" +
                         ("0" + utcDate.getMinutes()).slice(-2) + ":" +
                         ("0" + utcDate.getSeconds()).slice(-2);
  return localTimestamp;
}
  
export { convertToLocalTimestamp };