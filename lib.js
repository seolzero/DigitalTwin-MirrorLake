exports.isObject = (obj) => {
   return (
      obj?.constructor === {}.constructor ||
      obj?.constructor.toString().includes("TextRow")
   );
};

exports.tryJSONparse = (obj) => {
   try {
      messageObject = JSON.parse(obj);
      return messageObject;
   } catch (e) {
      return false;
   }
};
