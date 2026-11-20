import NSI
import Lean.Util.CollectAxioms

open Lean Elab Command

-- Inspect every declaration from a local proof module, including private helpers.
run_cmd do
  let env ← getEnv
  let allowed := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut count := 0
  for (name, info) in env.constants do
    let some idx := env.getModuleIdxFor? name | continue
    let moduleName := env.header.moduleNames[idx]!
    if moduleName == `NSI || (`NSI).isPrefixOf moduleName then
      if info.isUnsafe then
        throwError "Unsafe local declaration: {name}"
      let axioms ← collectAxioms name
      for ax in axioms do
        unless allowed.contains ax do
          throwError "Unapproved axiom in {name}: {ax}"
      let kind := if info matches .thmInfo _ then "theorem" else "definition_or_auxiliary"
      let row := Json.mkObj [
        ("name", toJson name.toString),
        ("module", toJson moduleName.toString),
        ("kind", toJson kind),
        ("axioms", toJson (axioms.map Name.toString))]
      logInfo m!"NSI_AUDIT {row.compress}"
      count := count + 1
  if count == 0 then
    throwError "No local proof declarations were checked."
  logInfo m!"NSI_AUDIT_COUNT {count}"
