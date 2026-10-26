import fs from 'node:fs/promises';
import {finalizePresentation} from 'file:///C:/Users/xujiayi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='D:/3. projects/xjy/research2workflow';
const skill='C:/Users/xujiayi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const target=root+'/output/2027年度计划_许佳宜_结合月报周报修订.pptx';
const candidate=root+'/.ppt_build/revision-candidate.pptx';
await fs.rename(target,candidate);
try {
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:target,pythonExecutable:'C:/Users/xujiayi/AppData/Local/Programs/Python/Python313/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000'],explicitTotalSlideCount:2,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],verifyArtifactToolImport:true,receiptPath:root+'/.ppt_build/revision-validation.json'});
console.log(JSON.stringify(result));
}catch(e){await fs.copyFile(candidate,target);throw e;}
